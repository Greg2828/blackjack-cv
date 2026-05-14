"""Guarda 3 versiones del mismo frame para identificar el orden de canales correcto."""
import cv2
from picamera2 import Picamera2
import numpy as np
from pathlib import Path

Path("data").mkdir(exist_ok=True)

cam = Picamera2()
cfg = cam.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
cam.configure(cfg)
cam.start()

# Descarta los primeros frames (la cámara necesita estabilizarse)
for _ in range(10):
    cam.capture_array("main")

frame = cam.capture_array("main")
cam.stop()
cam.close()

print(f"Shape: {frame.shape}  dtype: {frame.dtype}")
print(f"Pixel [240,320]: canal0={frame[240,320,0]}  canal1={frame[240,320,1]}  canal2={frame[240,320,2]}")

# Versión 1: sin ninguna conversión
cv2.imwrite("data/color_test_1_raw.jpg",          frame)
# Versión 2: convirtiendo como si fuera RGB → BGR
cv2.imwrite("data/color_test_2_rgb2bgr.jpg",      cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
# Versión 3: flip horizontal (por si está espejada)
cv2.imwrite("data/color_test_3_flip.jpg",         cv2.flip(frame, 1))

print()
print("Guardadas 3 fotos en data/:")
print("  color_test_1_raw.jpg       — sin conversión")
print("  color_test_2_rgb2bgr.jpg   — conversión RGB→BGR")
print("  color_test_3_flip.jpg      — espejo horizontal")
print()
print("Abre las 3 con:  eog data/color_test_*.jpg")
print("Dime cuál tiene los colores correctos (1, 2 o 3).")
