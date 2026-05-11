import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

"""
Bucle principal de Blackjack CV.

Controles durante la partida:
  q  — salir
  n  — nueva mano (guarda la actual si está resuelta)
  r  — resetear mano sin guardar
  +  — incrementar apuesta en 5
  -  — decrementar apuesta en 5
"""
import cv2

import config
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
    p_cards  = state.player_hand.visible_cards
    d_visible = state.dealer_hand.visible_cards

    if len(p_cards) < 2 or not d_visible:
        return Phase.WAITING_BET

    if state.player_hand.is_bust():
        return Phase.RESOLVED

    if state.dealer_hand.has_hidden:
        return Phase.PLAYER_TURN

    d_total = state.dealer_hand.total()
    if state.dealer_hand.is_bust() or d_total >= 17:
        return Phase.RESOLVED

    return Phase.DEALER_TURN


def main() -> None:
    logger   = HandLogger(config.LOG_FILE)
    display  = Display()
    motion   = MotionDetector(
        threshold=config.MOTION_THRESHOLD,
        stability_seconds=config.STABILITY_SECONDS,
    )
    detector = CardDetector(
        model_path=config.MODEL_PATH,
        card_classes=config.CARD_CLASSES,
        zone_dealer=config.ZONE_DEALER,
        zone_player=config.ZONE_PLAYER,
    )
    chip_detector = ChipDetector(
        chip_values=config.CHIP_VALUES,
        chip_hsv_ranges=config.CHIP_HSV_RANGES,
        min_area=config.CHIP_MIN_AREA,
    )

    if not detector.ready:
        print("[AVISO] Modelo YOLOv8 no encontrado.")
        print(f"        Genera datos con scripts/generate_synthetic_data.py")
        print(f"        y entrena el modelo. Colócalo en: {config.MODEL_PATH}")

    state = GameState(bankroll=config.STARTING_BANKROLL, bet=10.0)
    recommended_sequence: list[Action] = []
    taken_sequence:       list[Action] = []
    resolved_this_hand = False

    print("Blackjack CV arrancado.")
    print("  [q] salir   [n] nueva mano   [r] reset   [+/-] apuesta")

    with Camera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT) as cam:
        while True:
            frame = cam.read()
            _, just_stabilized = motion.update(frame)

            # Re-detectamos cuando la escena se estabiliza tras movimiento.
            # Esto evita procesar frames ruidosos mientras el jugador mueve cartas.
            if just_stabilized:
                if detector.ready:
                    player_cards, dealer_cards = detector.detect(frame)
                    state.player_hand = Hand(player_cards)
                    state.dealer_hand = Hand(dealer_cards)
                    state.phase       = _infer_phase(state)

                # Detección de fichas en zona de apuesta (solo si calibrado y esperando apuesta)
                if chip_detector.calibrated and state.phase == Phase.WAITING_BET:
                    detected_bet = chip_detector.detect(
                        frame,
                        config.ZONE_BETTING['y_min'],
                        config.ZONE_BETTING['y_max'],
                    )
                    if detected_bet > 0:
                        state.bet = detected_bet

            recommendation: Action | None = None

            if state.phase == Phase.PLAYER_TURN:
                resolved_this_hand = False
                dealer_upcard  = state.dealer_hand.visible_cards[0]
                first_decision = len(state.player_hand.visible_cards) == 2
                can_split_now = first_decision and state.player_hand.is_pair()
                recommendation = recommend(
                    state.player_hand,
                    dealer_upcard,
                    can_split=can_split_now,
                    can_double=first_decision,
                    can_surrender=first_decision,
                )
                row = full_row(
                    state.player_hand,
                    can_split=can_split_now,
                    can_double=first_decision,
                    can_surrender=first_decision,
                )
                if not recommended_sequence or recommended_sequence[-1] != recommendation:
                    recommended_sequence.append(recommendation)
                display.show(state, recommendation,
                             strategy_row=row,
                             dealer_upcard_rank=dealer_upcard.rank)

            elif state.phase == Phase.RESOLVED and not resolved_this_hand:
                outcome, delta = state.resolve()
                state.bankroll += delta
                display.show_outcome(outcome, delta)
                resolved_this_hand = True

            else:
                display.show(state, recommendation)

            cv2.imshow("Camara", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                if resolved_this_hand:
                    outcome, delta = state.resolve()
                    logger.log(state, recommended_sequence, taken_sequence, outcome, delta)
                recommended_sequence = []
                taken_sequence       = []
                resolved_this_hand   = False
                motion.reset()
                state = GameState(bankroll=state.bankroll, bet=state.bet)
            elif key == ord('r'):
                recommended_sequence = []
                taken_sequence       = []
                resolved_this_hand   = False
                motion.reset()
                state = GameState(bankroll=state.bankroll, bet=state.bet)
            elif key == ord('+') and state.phase == Phase.WAITING_BET:
                state.bet = max(5.0, state.bet + 5.0)
            elif key == ord('-') and state.phase == Phase.WAITING_BET:
                state.bet = max(5.0, state.bet - 5.0)

    display.close()
    cv2.destroyAllWindows()
    print("Sistema cerrado.")


if __name__ == '__main__':
    main()
