import numpy as np
import cv2


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
        self.chip_values = chip_values
        self.chip_hsv_ranges = chip_hsv_ranges
        self.min_area = min_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    @property
    def calibrated(self) -> bool:
        return any(
            v.get('lower') is not None and v.get('upper') is not None
            for v in self.chip_hsv_ranges.values()
        )

    def detect(self, frame: np.ndarray, zone_y_min: float, zone_y_max: float) -> float:
        """Devuelve el valor total de fichas detectadas en la zona indicada.
        Retorna 0.0 si los rangos no están calibrados o no hay fichas."""
        if not self.calibrated:
            return 0.0

        h = frame.shape[0]
        y1 = int(h * zone_y_min)
        y2 = int(h * zone_y_max)
        roi = frame[y1:y2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        total = 0.0
        for chip_name, value in self.chip_values.items():
            ranges = self.chip_hsv_ranges.get(chip_name, {})
            lower = ranges.get('lower')
            upper = ranges.get('upper')
            if lower is None or upper is None:
                continue

            mask = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                # Solo contornos suficientemente circulares (fichas son redondas)
                if 4 * np.pi * area / (perimeter ** 2) > 0.65:
                    total += value

        return total
