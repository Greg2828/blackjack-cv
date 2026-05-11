# Blackjack CV

Sistema de visión artificial sobre Raspberry Pi 5 que observa una mesa real de blackjack, recomienda la jugada óptima según basic strategy, y registra cada partida para análisis estadístico posterior.

**Hardware**: Raspberry Pi 5 + Pi Camera
**Stack**: Python · YOLOv8 · OpenCV · NumPy · Pandas

## Estado: en desarrollo (día 1 de 7)

## Setup

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest tests/ -v
```
