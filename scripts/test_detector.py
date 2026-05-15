"""
ARCHIVO: scripts/test_detector.py
PROPÓSITO: Verifica que el modelo YOLOv8 detecta las cartas correctamente
           en tiempo real, mostrando el feed de la cámara en una ventana.

ARQUITECTURA (importante — explica por qué este script es complejo):
  En la Raspberry Pi 5 con Wayland, la combinación
      picamera2 (GLib) + torch + cv2.imshow (Qt5)
  hace que la ventana de OpenCV salga en blanco (ver README §13).
  Solución aplicada aquí:
    1. picamera2 corre en un SUBPROCESO (src/perception/camera_ipc.py),
       aislándolo del proceso principal — fuera está GLib.
    2. La ventana se hace con TKINTER en lugar de cv2.imshow — fuera está Qt5.
  Resultado: el proceso principal solo tiene torch + tkinter + numpy + cv2
  (para dibujar sobre el frame, NO para crear ventanas).

Uso:
    python scripts/test_detector.py

Controles:
    ESPACIO       → detectar cartas en el frame actual (congela imagen)
    cualquier otra tecla cuando está congelado → volver al feed en vivo
    s             → guardar captura en data/
    q  o  ESC     → salir
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

import config
from src.perception.camera_ipc import CameraIPC as Camera

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _box_color(conf):
    """Color del recuadro según la confianza de YOLO (verde alta, rojo baja)."""
    if conf >= 0.80:
        return (0, 210, 0)
    if conf >= 0.50:
        return (0, 165, 255)
    return (0, 0, 220)


def _draw_zones(frame):
    """Dibuja las líneas que separan las zonas dealer / player / betting."""
    h, w = frame.shape[:2]
    for label, zone, color in [
        ('DEALER',  config.ZONE_DEALER,  (0, 100, 255)),
        ('PLAYER',  config.ZONE_PLAYER,  (0, 210,   0)),
        ('BETTING', config.ZONE_BETTING, (255, 100,  0)),
    ]:
        y1 = int(h * zone['y_min'])
        cv2.line(frame, (0, y1), (w, y1), color, 1)
        cv2.putText(frame, label, (8, y1 + 22), _FONT, 0.65, color, 2)


def _run_yolo(frame, model):
    """Pasa el frame por YOLO y devuelve el frame anotado con las detecciones."""
    h, w = frame.shape[:2]
    out = frame.copy()
    _draw_zones(out)

    t0 = time.monotonic()
    results = model(frame, verbose=False)[0]
    ms = (time.monotonic() - t0) * 1000

    d_max = int(h * config.ZONE_DEALER['y_max'])
    p_min = int(h * config.ZONE_PLAYER['y_min'])
    p_max = int(h * config.ZONE_PLAYER['y_max'])
    dealer_cards, player_cards = [], []

    for box in results.boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        rank = config.CARD_CLASSES[cls_idx]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yc = (y1 + y2) / 2
        color = _box_color(conf)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{rank} {conf*100:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.75, 2)
        ly = max(y1 - 8, 20)
        cv2.rectangle(out, (x1, ly - lh - 4), (x1 + lw + 6, ly + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, ly), _FONT, 0.75, (0, 0, 0), 2)

        if yc < d_max:
            dealer_cards.append(f"{rank}({conf*100:.0f}%)")
        elif p_min <= yc < p_max:
            player_cards.append(f"{rank}({conf*100:.0f}%)")

    summary = (f"DEALER: {', '.join(dealer_cards) or '--'}   "
               f"PLAYER: {', '.join(player_cards) or '--'}   [{ms:.0f}ms]")
    cv2.rectangle(out, (0, h - 38), (w, h), (20, 20, 20), -1)
    cv2.putText(out, summary, (10, h - 12), _FONT, 0.58, (200, 200, 200), 1)
    cv2.rectangle(out, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(out, "Pulsa cualquier tecla = feed en vivo   s = guardar   q = salir",
                (10, 24), _FONT, 0.58, (160, 160, 160), 1)

    print(f"  {ms:.0f}ms | DEALER: {dealer_cards or ['--']} | PLAYER: {player_cards or ['--']}")
    return out


class DetectorApp:
    """Ventana Tkinter que muestra el feed de la cámara con detecciones YOLO."""

    def __init__(self, root, cam, model, display_scale=0.75):
        self.root = root
        self.cam = cam
        self.model = model
        self.frozen = None  # cuando hay detección congelada, contiene el frame anotado
        self.display_scale = display_scale  # 0.75 = 960×540 en pantalla (1280×720 → más legible)

        self.last_t = time.monotonic()
        self.frame_count = 0
        self.fps = 0.0

        root.title("Test Detector — YOLOv8 (Tkinter)")
        # Color negro de fondo para que se note si algo falla
        root.configure(bg='black')

        # Label que contendrá la imagen.
        self.image_label = tk.Label(root, bg='black')
        self.image_label.pack()

        # Barra de estado debajo
        self.status_var = tk.StringVar(
            value="ESPACIO = detectar    s = guardar    q / ESC = salir"
        )
        tk.Label(root, textvariable=self.status_var,
                 font=("DejaVu Sans", 12), bg='black', fg='white',
                 pady=6).pack(fill='x')

        # Bindings de teclado. Importante: bind al root para capturar
        # las pulsaciones independientemente del foco interno.
        root.bind('<space>', self._on_space)
        root.bind('s', self._on_save)
        root.bind('S', self._on_save)
        root.bind('q', self._on_quit)
        root.bind('Q', self._on_quit)
        root.bind('<Escape>', self._on_quit)
        root.bind('<Key>', self._on_any_key)  # cualquier otra tecla → unfreeze
        root.protocol("WM_DELETE_WINDOW", self._on_quit)

        # Empezamos el bucle de refresco.
        self._tick()

    def _tick(self):
        """Lee un frame, lo dibuja en la ventana, y se reagenda a sí mismo."""
        try:
            if self.frozen is not None:
                # Estado "congelado": mostramos el frame con detecciones.
                display = self.frozen
            else:
                frame = self.cam.read()
                display = frame.copy()
                _draw_zones(display)

                # FPS
                self.frame_count += 1
                elapsed = time.monotonic() - self.last_t
                if elapsed >= 0.5:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_t = time.monotonic()

                h, w = display.shape[:2]
                cv2.rectangle(display, (0, h - 38), (w, h), (20, 20, 20), -1)
                cv2.putText(display,
                            f"FPS: {self.fps:4.1f}   ESPACIO=detectar  s=guardar  q=salir",
                            (10, h - 12), _FONT, 0.6, (200, 200, 200), 1)

            self._show(display)
        except Exception as e:
            print(f"Error en _tick: {e}")

        # 30 ms = ~33 Hz de refresco. La cámara va a ~30 FPS también.
        self.root.after(30, self._tick)

    def _show(self, bgr_frame):
        """Convierte un frame numpy BGR a PhotoImage de Tkinter y lo muestra."""
        # OpenCV usa BGR, PIL usa RGB → conversión obligatoria.
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Escalar para que la ventana quepa en pantalla con holgura.
        if self.display_scale != 1.0:
            h, w = rgb.shape[:2]
            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)

        # Tkinter borra la imagen si no se guarda una referencia explícita;
        # por eso self.image_label.image = photo (no es un typo).
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def _on_space(self, event):
        """SPACE: si está en vivo → congela con detecciones. Si está congelado → vuelve a vivo."""
        if self.frozen is None:
            frame = self.cam.read()
            self.frozen = _run_yolo(frame, self.model)
        else:
            self.frozen = None
        return 'break'  # evita que <Key> también se dispare

    def _on_save(self, event):
        """Guarda el frame actual (congelado o en vivo) en data/."""
        Path("data").mkdir(exist_ok=True)
        path = f"data/detector_test_{int(time.time())}.jpg"
        if self.frozen is not None:
            cv2.imwrite(path, self.frozen)
        else:
            frame = self.cam.read()
            _draw_zones(frame)
            cv2.imwrite(path, frame)
        print(f"  Guardado: {path}")
        return 'break'

    def _on_quit(self, event=None):
        self.root.quit()
        return 'break'

    def _on_any_key(self, event):
        """Cualquier tecla no mapeada arriba sale del estado congelado."""
        if self.frozen is not None and event.keysym not in (
                'space', 's', 'S', 'q', 'Q', 'Escape'):
            self.frozen = None


def main():
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        print(f"ERROR: modelo no encontrado en {model_path}")
        sys.exit(1)

    Path("data").mkdir(exist_ok=True)

    # 1. Abrir la cámara (subproceso, no carga picamera2 en este proceso).
    print("Abriendo cámara (subproceso picamera2)...")
    with Camera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT) as cam:
        # Warmup: descarta los primeros frames para que auto-exposición y
        # auto-foco se estabilicen.
        for _ in range(20):
            cam.read()

        # 2. Cargar el modelo YOLO (esto sí pasa en el proceso principal).
        print("Cargando modelo YOLO (puede tardar 1-2 min en Pi 5)...")
        model = YOLO(str(model_path))
        print("Listo. Abriendo ventana Tkinter...")

        # 3. Crear ventana Tkinter y arrancar el bucle de refresco.
        root = tk.Tk()
        app = DetectorApp(root, cam, model)
        try:
            root.mainloop()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass

    print("Cerrado.")


if __name__ == '__main__':
    main()
