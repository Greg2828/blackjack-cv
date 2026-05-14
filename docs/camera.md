# Cámara — Raspberry Pi Camera Module 3 (IMX708)

## Hardware

| Propiedad         | Valor                          |
|-------------------|--------------------------------|
| Modelo sensor     | IMX708                         |
| Resolución máxima | 4608 × 2592 px (12.3 MP)       |
| Tamaño de píxel   | 1.4 µm × 1.4 µm               |
| Rotación física   | 180° (propiedad del módulo)    |
| Autofoco          | Continuo (Phase Detection AF)  |
| Pipeline          | libcamera / PiSP (Pi 5)        |

## Modos del sensor

| Modo | Resolución   | FPS máx | Formato raw     |
|------|--------------|---------|-----------------|
| 0    | 1536 × 864   | 120.1   | SRGGB10_CSI2P   |
| 1    | 2304 × 1296  | 56.0    | SRGGB10_CSI2P   |
| 2    | 4608 × 2592  | 14.3    | SRGGB10_CSI2P   |

En el proyecto usamos **1280 × 720** vía escalado del modo 0 (suficiente resolución, alta fluidez).

## Configuración en `src/perception/camera.py`

### Por qué picamera2 y no cv2.VideoCapture

En Raspberry Pi 5 con Raspberry Pi OS Bookworm, la Pi Camera usa el stack **libcamera**.
`cv2.VideoCapture` abre el dispositivo pero no puede leer frames (`read()` falla).
La solución es usar **picamera2**, preinstalada en el sistema, accesible desde el venv
creado con `--system-site-packages`.

### Por qué format RGB888 sin conversión

En Pi 5, `picamera2` devuelve los datos en orden **B-G-R** aunque el formato configurado
sea `RGB888`. Se verificó empíricamente: el frame raw es ya BGR y OpenCV lo muestra
con colores correctos sin ninguna conversión. Aplicar `cv2.COLOR_RGB2BGR` empeora los colores.

### Por qué no se aplica la rotación de 180°

El sensor reporta `Rotation: 180` como metadato de hardware, pero **picamera2 no aplica
esa rotación automáticamente** (el campo `transform` de la configuración es `identity`).
La cámara está montada de forma que la imagen sale con la orientación correcta sin rotar,
así que no se aplica ninguna corrección en el código.

### Controles activados al arrancar

```python
cam.set_controls({
    "AfMode":   2,     # Autofoco continuo (0=manual, 1=auto, 2=continuo)
    "AfRange":  0,     # Rango normal      (0=normal, 1=macro, 2=full)
    "AfSpeed":  1,     # Respuesta rápida  (0=normal, 1=fast)
    "AeEnable": True,  # Exposición automática
    "AwbEnable": True, # Balance de blancos automático
})
```

**AfMode=2 (continuo)** es importante para el dataset: garantiza que las cartas salgan
nítidas sin necesidad de esperar o pulsar un botón de enfoque.

## Controles disponibles (referencia)

Útiles si en el futuro se necesita ajuste manual:

| Control          | Min    | Max         | Default  | Descripción                          |
|------------------|--------|-------------|----------|--------------------------------------|
| ExposureTime     | 26 µs  | 220 s       | 20000 µs | Tiempo de exposición manual          |
| AnalogueGain     | 1.12   | 16.0        | 1.0      | Ganancia ISO analógica               |
| Brightness       | -1.0   | 1.0         | 0.0      | Brillo general                       |
| Contrast         | 0.0    | 32.0        | 1.0      | Contraste                            |
| Saturation       | 0.0    | 32.0        | 1.0      | Saturación de color                  |
| Sharpness        | 0.0    | 16.0        | 1.0      | Nitidez                              |
| AwbMode          | 0      | 7           | auto     | Modo balance blancos (auto/tungsten/…)|
| LensPosition     | 0.0    | 15.0        | 1.0      | Posición del foco (modo manual)      |
| HdrMode          | 0      | 4           | 0        | HDR (0=off)                          |
| ExposureValue    | -8.0   | 8.0         | 0.0      | Compensación de exposición (EV)      |
