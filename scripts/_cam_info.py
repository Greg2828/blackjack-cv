import json
from picamera2 import Picamera2

cam = Picamera2()

print("=== PROPIEDADES DEL SENSOR ===")
for k, v in cam.camera_properties.items():
    print(f"  {k}: {v}")

print("\n=== MODOS DEL SENSOR ===")
for i, m in enumerate(cam.sensor_modes):
    size = m['size']
    fps  = m['fps']
    fmt  = m['format']
    print(f"  Modo {i}: size={size}  fps={fps:.1f}  format={fmt}")

print("\n=== CONTROLES DISPONIBLES (min, max, default) ===")
for k, v in sorted(cam.camera_controls.items()):
    print(f"  {k}: {v}")

print("\n=== CONFIGURACION VIDEO 1280x720 ===")
cfg = cam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
cam.configure(cfg)
print(json.dumps(cfg, default=str, indent=2))

cam.close()
