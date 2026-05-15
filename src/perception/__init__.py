# Vacío a propósito: cada cliente importa el submódulo concreto que necesita
# (`from src.perception.camera import Camera`, `from src.perception.camera_ipc
# import CameraIPC`, etc.). Reexportar `Camera` aquí cargaría `picamera2`
# en cualquier proceso que importe el paquete, rompiendo el aislamiento de
# CameraIPC con respecto al bug GLib+Qt5+torch (README §13).
