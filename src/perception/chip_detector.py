"""
ARCHIVO: src/perception/chip_detector.py
PROPÓSITO: Detecta fichas de casino en la imagen usando análisis de color HSV.
           A diferencia de las cartas (que usa IA/YOLO), las fichas se detectan
           por su COLOR: cada tipo de ficha tiene un color distinto y uniforme.

           El proceso es:
           1. Recortar la zona de apuestas (franja inferior del frame)
           2. Convertir la imagen a espacio de color HSV
           3. Buscar píxeles dentro del rango de color de cada tipo de ficha
           4. Encontrar contornos circulares suficientemente grandes
           5. Sumar el valor de cada ficha encontrada

CÓMO SE CONECTA:
  - main.py crea un ChipDetector y llama a detect() cuando la escena se estabiliza
  - config.py provee CHIP_VALUES, CHIP_HSV_RANGES y CHIP_MIN_AREA
  - calibrate_chips.py produce los valores HSV que se pegan en config.py

NOTA: hasta que calibres las fichas con calibrate_chips.py,
      detect() siempre devuelve 0.0 sin producir errores.
"""

import numpy as np  # arrays numéricos (imágenes = arrays de píxeles)
import cv2          # OpenCV: procesamiento de imagen


class ChipDetector:
    """Detecta fichas en la zona de apuesta usando rangos de color HSV.

    Devuelve el valor total de las fichas detectadas en la zona de betting.
    Si los rangos no están calibrados (None), detect() devuelve 0.0 sin error.
    Usa scripts/calibrate_chips.py para obtener los rangos HSV de tus fichas.
    """

    def __init__(
        self,
        chip_values: dict[str, int],
        chip_hsv_ranges: dict,
        min_area: int = 500,
    ):
        """Constructor.

        Parámetros:
          chip_values: {'chip_1': 1, 'chip_2': 5, 'chip_3': 25} (de config.py)
          chip_hsv_ranges: rangos HSV por tipo de ficha (de config.py)
          min_area: área mínima en px² para considerar un contorno como ficha
        """
        self.chip_values    = chip_values
        self.chip_hsv_ranges = chip_hsv_ranges
        self.min_area       = min_area

        # Kernel morfológico elíptico 5×5.
        # Se usa para "limpiar" la máscara de color:
        # eliminar píxeles aislados de ruido y cerrar pequeños huecos en la ficha.
        # cv2.MORPH_ELLIPSE crea una forma elíptica, ideal para objetos circulares.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    @property
    def calibrated(self) -> bool:
        """True si al menos UN tipo de ficha tiene rangos HSV definidos (no None).
        Si es False, no tiene sentido intentar detectar fichas."""
        return any(
            v.get('lower') is not None and v.get('upper') is not None
            for v in self.chip_hsv_ranges.values()
        )

    def detect(self, frame: np.ndarray, zone_y_min: float, zone_y_max: float) -> float:
        """Detecta fichas en la franja vertical indicada del frame.

        Parámetros:
          frame: imagen BGR completa de la cámara
          zone_y_min, zone_y_max: fracciones de la altura que delimitan la zona de apuestas
                                   (de config.ZONE_BETTING)

        Devuelve: valor total de las fichas detectadas (float, en euros).
                  0.0 si no hay fichas, si no están calibradas, o si no se detecta nada.
        """

        # Sin calibración no podemos saber de qué color es cada ficha.
        if not self.calibrated:
            return 0.0

        # Recortamos la zona de apuestas del frame completo.
        h  = frame.shape[0]           # altura total del frame en píxeles
        y1 = int(h * zone_y_min)      # píxel de inicio de la zona (ej: 75% de 720 = 540)
        y2 = int(h * zone_y_max)      # píxel de fin de la zona    (ej: 100% de 720 = 720)
        roi = frame[y1:y2]            # ROI = Region Of Interest (recorte de la imagen)

        # Convertimos el recorte de BGR a HSV.
        # HSV (Hue, Saturation, Value) es mucho más robusto que BGR para detectar colores:
        # si cambia la iluminación, H (el tono del color) apenas varía,
        # mientras que en BGR los tres canales cambian drásticamente.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        total = 0.0  # acumulador del valor total de fichas encontradas

        # Procesamos cada tipo de ficha por separado.
        for chip_name, value in self.chip_values.items():
            ranges = self.chip_hsv_ranges.get(chip_name, {})
            lower  = ranges.get('lower')  # límite inferior del rango HSV [H, S, V]
            upper  = ranges.get('upper')  # límite superior del rango HSV [H, S, V]

            # Si esta ficha no está calibrada, la saltamos.
            if lower is None or upper is None:
                continue

            # cv2.inRange() crea una máscara binaria:
            #   píxel BLANCO (255) si está dentro del rango [lower, upper]
            #   píxel NEGRO  (0)   si está fuera
            mask = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),  # convertimos la lista a array numpy
                np.array(upper, dtype=np.uint8),
            )

            # MORPH_OPEN (erosión + dilatación): elimina pequeños puntos de ruido.
            # Cualquier mancha blanca más pequeña que el kernel (5×5) desaparece.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

            # MORPH_CLOSE (dilatación + erosión): cierra pequeños agujeros dentro de la ficha.
            # Une partes blancas que quedaron separadas por sombras o reflejos.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

            # Buscamos los contornos (bordes de las regiones blancas) en la máscara.
            # cv2.RETR_EXTERNAL: solo contornos externos (sin contornos dentro de otros).
            # cv2.CHAIN_APPROX_SIMPLE: comprime el contorno guardando solo puntos clave.
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Evaluamos cada contorno encontrado.
            for cnt in contours:
                area = cv2.contourArea(cnt)  # área en píxeles² del contorno

                # Filtramos objetos demasiado pequeños (ruido, reflejos, botones...).
                if area < self.min_area:
                    continue

                # Calculamos el perímetro del contorno.
                # True = el contorno está cerrado (forma una forma completa).
                perimeter = cv2.arcLength(cnt, True)

                if perimeter == 0:
                    continue  # evita división por cero en contornos degenerados

                # PRUEBA DE CIRCULARIDAD: 4π×area / perímetro² = 1.0 para un círculo perfecto.
                # Para una ficha real (casi circular), debería ser > 0.65.
                # Un cuadrado tiene circularidad ~0.78, un rectángulo muy alargado < 0.3.
                # Esta prueba filtra objetos rectangulares (cartas, mesas) que podrían
                # tener el color equivocado.
                if 4 * np.pi * area / (perimeter ** 2) > 0.65:
                    total += value  # ¡encontramos una ficha! añadimos su valor

        return total
