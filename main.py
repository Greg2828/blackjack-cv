import os
# Estas dos líneas fuerzan a OpenCV a usar un backend gráfico compatible con la Pi.
# QT_QPA_PLATFORM='xcb' evita que OpenCV intente usar Wayland (que a veces da error).
# QT_LOGGING_RULES suprime mensajes de depuración de Qt que llenarían el terminal.
# setdefault significa "ponlo solo si no está ya definido".
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

"""
ARCHIVO: main.py
PROPÓSITO: Punto de entrada del sistema con cámara real.
           Es el "director de orquesta": crea todos los módulos,
           conecta la cámara con la detección, la detección con la estrategia,
           y la estrategia con la pantalla.

           Lo ejecutas con:  python main.py

Controles durante la partida:
  q  — salir
  n  — nueva mano (guarda la actual si está resuelta)
  r  — resetear mano sin guardar
  +  — incrementar apuesta en 5
  -  — decrementar apuesta en 5

FLUJO PRINCIPAL (bucle infinito):
  1. Leer frame de la cámara
  2. Detector de movimiento: ¿se estabilizó la escena?
     → Si sí: ejecutar YOLO + detección de fichas
  3. Según la fase actual: mostrar recomendación o resultado
  4. Mostrar el frame de la cámara en otra ventana
  5. Leer tecla pulsada y actuar en consecuencia
  6. Repetir hasta que el usuario pulse 'q'
"""
import cv2   # OpenCV: para mostrar el feed de la cámara y leer teclas

# Importamos la configuración central del proyecto.
import config

# Importamos los módulos que hemos construido en src/.
from src.game.state import GameState, Phase, Action, Outcome
from src.game.hand import Hand
from src.perception.camera import Camera
from src.perception.detector import CardDetector
from src.perception.chip_detector import ChipDetector
from src.core.motion import MotionDetector
from src.decision.strategy import recommend, full_row
from src.ui.display import Display
from src.analysis.logger import HandLogger


def _infer_phase(state: GameState) -> Phase:
    """Determina la fase actual de la mano mirando el estado de las cartas.
    Se llama cada vez que detectamos nuevas cartas para actualizar state.phase.

    Lógica de inferencia:
      - Menos de 2 cartas del jugador O sin cartas del crupier → WAITING_BET
      - El jugador está bust → RESOLVED
      - El crupier aún tiene carta tapada (has_hidden) → PLAYER_TURN (el jugador decide)
      - El crupier tiene 17+ (o bust) → RESOLVED
      - Si no → DEALER_TURN (el crupier pide más cartas)
    """
    p_cards   = state.player_hand.visible_cards   # cartas visibles del jugador
    d_visible = state.dealer_hand.visible_cards   # cartas visibles del crupier

    # Sin cartas suficientes para determinar la fase: todavía esperando el reparto.
    if len(p_cards) < 2 or not d_visible:
        return Phase.WAITING_BET

    # Si el jugador se pasó de 21, la mano ya terminó.
    if state.player_hand.is_bust():
        return Phase.RESOLVED

    # Si el crupier todavía tiene una carta tapada, es el turno del jugador.
    # (El sistema de visión ve el BACK = dorso de la carta tapada del crupier).
    if state.dealer_hand.has_hidden:
        return Phase.PLAYER_TURN

    # En este punto: no hay carta tapada → el crupier ya mostró todo.
    d_total = state.dealer_hand.total()

    # Si el crupier se pasó o tiene 17+, la mano está resuelta.
    if state.dealer_hand.is_bust() or d_total >= 17:
        return Phase.RESOLVED

    # Si no llegó a 17, el crupier debe seguir pidiendo cartas.
    return Phase.DEALER_TURN


def main() -> None:
    """Función principal: inicializa todos los módulos y ejecuta el bucle de la partida."""

    # ── Inicialización de módulos ──────────────────────────────────────────────

    # HandLogger: guarda cada mano en el CSV al final.
    logger = HandLogger(config.LOG_FILE)

    # Display: la ventana de interfaz con la recomendación y tabla.
    display = Display()

    # MotionDetector: detecta cuando el jugador deja de mover las cartas.
    motion = MotionDetector(
        threshold=config.MOTION_THRESHOLD,
        stability_seconds=config.STABILITY_SECONDS,
    )

    # CardDetector: carga el modelo YOLOv8 si existe y detecta cartas en el frame.
    detector = CardDetector(
        model_path=config.MODEL_PATH,
        card_classes=config.CARD_CLASSES,
        zone_dealer=config.ZONE_DEALER,
        zone_player=config.ZONE_PLAYER,
    )

    # ChipDetector: detecta fichas de casino por color HSV.
    chip_detector = ChipDetector(
        chip_values=config.CHIP_VALUES,
        chip_hsv_ranges=config.CHIP_HSV_RANGES,
        min_area=config.CHIP_MIN_AREA,
    )

    # Advertimos si el modelo YOLO todavía no existe (hay que entrenarlo primero).
    if not detector.ready:
        print("[AVISO] Modelo YOLOv8 no encontrado.")
        print(f"        Genera datos con scripts/generate_synthetic_data.py")
        print(f"        y entrena el modelo. Colócalo en: {config.MODEL_PATH}")

    # ── Estado inicial de la partida ───────────────────────────────────────────

    # GameState: el estado completo de la mano actual.
    state = GameState(bankroll=config.STARTING_BANKROLL, bet=10.0)

    # Guardamos las secuencias de acciones para el log al final de la mano.
    recommended_sequence: list[Action] = []  # qué recomendó la estrategia
    taken_sequence:       list[Action] = []  # qué hizo el jugador (aún no implementado con teclas)

    # Evita que resolve() se llame dos veces por la misma mano.
    resolved_this_hand = False

    print("Blackjack CV arrancado.")
    print("  [q] salir   [n] nueva mano   [r] reset   [+/-] apuesta")

    # ── Bucle principal ────────────────────────────────────────────────────────

    # 'with Camera(...) as cam:' abre la cámara y garantiza que se cierre al salir.
    with Camera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT) as cam:
        while True:
            # 1. Leemos el siguiente frame de la cámara.
            frame = cam.read()

            # 2. Actualizamos el detector de movimiento.
            # just_stabilized = True solo en el primer frame quieto tras movimiento.
            _, just_stabilized = motion.update(frame)

            # 3. Si la escena acaba de estabilizarse, ejecutamos la detección.
            # Esto evita correr YOLO en cada frame (sería demasiado lento).
            if just_stabilized:
                if detector.ready:
                    # Detectamos las cartas y actualizamos el estado.
                    player_cards, dealer_cards = detector.detect(frame)
                    state.player_hand = Hand(player_cards)
                    state.dealer_hand = Hand(dealer_cards)
                    # Inferimos la fase según lo que detectamos.
                    state.phase = _infer_phase(state)

                # Detección de fichas: solo si están calibradas y esperamos la apuesta.
                if chip_detector.calibrated and state.phase == Phase.WAITING_BET:
                    detected_bet = chip_detector.detect(
                        frame,
                        config.ZONE_BETTING['y_min'],
                        config.ZONE_BETTING['y_max'],
                    )
                    # Si se detectaron fichas con valor > 0, actualizamos la apuesta.
                    if detected_bet > 0:
                        state.bet = detected_bet

            # 4. Procesamos la fase actual y mostramos la interfaz.
            recommendation: Action | None = None

            if state.phase == Phase.PLAYER_TURN:
                # Es el turno del jugador: calculamos y mostramos la recomendación.
                resolved_this_hand = False
                dealer_upcard  = state.dealer_hand.visible_cards[0]
                first_decision = len(state.player_hand.visible_cards) == 2
                can_split_now  = first_decision and state.player_hand.is_pair()

                # Pedimos la recomendación a la basic strategy.
                recommendation = recommend(
                    state.player_hand,
                    dealer_upcard,
                    can_split=can_split_now,
                    can_double=first_decision,
                    can_surrender=first_decision,
                )

                # Pedimos la fila completa para la tabla visual (10 celdas).
                row = full_row(
                    state.player_hand,
                    can_split=can_split_now,
                    can_double=first_decision,
                    can_surrender=first_decision,
                )

                # Solo añadimos a la secuencia si la recomendación cambió.
                # (Evita duplicados si el frame se analiza varias veces con las mismas cartas).
                if not recommended_sequence or recommended_sequence[-1] != recommendation:
                    recommended_sequence.append(recommendation)

                display.show(state, recommendation,
                             strategy_row=row,
                             dealer_upcard_rank=dealer_upcard.rank)

            elif state.phase == Phase.RESOLVED and not resolved_this_hand:
                # La mano acaba de resolverse por primera vez: calculamos resultado.
                outcome, delta = state.resolve()
                state.bankroll += delta
                display.show_outcome(outcome, delta)
                resolved_this_hand = True

            else:
                # Cualquier otra fase (esperando apuesta, turno del crupier...):
                # mostramos la interfaz sin recomendación.
                display.show(state, recommendation)

            # 5. Mostramos el feed de la cámara en una ventana separada.
            cv2.imshow("Camara", frame)

            # 6. Esperamos hasta 30ms una tecla. Si el usuario no pulsa nada,
            # cv2.waitKey devuelve -1. '& 0xFF' extrae solo el último byte
            # (necesario en algunos sistemas para obtener el código ASCII correcto).
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                # Salir del bucle y cerrar el programa.
                break

            elif key == ord('n'):
                # Nueva mano: guardamos la mano actual en el CSV si ya se resolvió.
                if resolved_this_hand:
                    outcome, delta = state.resolve()
                    logger.log(state, recommended_sequence, taken_sequence, outcome, delta)

                # Reiniciamos las secuencias y el estado para la nueva mano.
                recommended_sequence = []
                taken_sequence       = []
                resolved_this_hand   = False
                motion.reset()   # forzamos re-detección cuando la nueva escena se estabilice
                # Creamos un nuevo GameState conservando el bankroll y la apuesta actual.
                state = GameState(bankroll=state.bankroll, bet=state.bet)

            elif key == ord('r'):
                # Resetear sin guardar (para corregir errores de detección).
                recommended_sequence = []
                taken_sequence       = []
                resolved_this_hand   = False
                motion.reset()
                state = GameState(bankroll=state.bankroll, bet=state.bet)

            elif key == ord('+') and state.phase == Phase.WAITING_BET:
                # Subir la apuesta en 5€ (solo mientras esperamos la apuesta).
                state.bet = max(5.0, state.bet + 5.0)

            elif key == ord('-') and state.phase == Phase.WAITING_BET:
                # Bajar la apuesta en 5€ (mínimo 5€).
                state.bet = max(5.0, state.bet - 5.0)

    # ── Limpieza al salir ──────────────────────────────────────────────────────

    display.close()
    cv2.destroyAllWindows()   # cierra todas las ventanas de OpenCV
    print("Sistema cerrado.")


# Esta condición asegura que main() solo se ejecute cuando corres este archivo directamente.
# Si otro archivo importa main.py, la función main() no se ejecuta automáticamente.
# Es la forma estándar de Python de definir el "punto de entrada" de un programa.
if __name__ == '__main__':
    main()
