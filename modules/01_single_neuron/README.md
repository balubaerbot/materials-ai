# Modul 1: Das einfachste Neuron

**Lernziel:** Verstehen, was ein einzelnes künstliches Neuron mathematisch ist
(lineare Regression: `y = w*x + b`) — und wo seine Grenzen liegen.

## Szenario

Wir fitten eine synthetische Spannungs-Dehnungs-Kurve von Stahl:
- Elastischer Bereich (Hooke'sches Gesetz) — **linear**, ein Neuron kann das perfekt.
- Plastischer Verfestigungsbereich — **nichtlinear**, ein Neuron kann das nur mitteln.

Genau dieser Kontrast ist die Lektion: Modul 2 wird zeigen, wie mehrere
Neuronen + Aktivierungsfunktionen (= richtiges neuronales Netz) diese
Nichtlinearität abbilden können.

## Starten

```bash
docker compose up module-01-single-neuron
```

Dann im Browser: `http://<host>:8501`

## Dateien

- `data_generator.py` — erzeugt die synthetische Werkstoffkennlinie
- `model.py` — das Einzel-Neuron (PyTorch) + Trainingsloop
- `app.py` — Streamlit-Dashboard mit Live-Visualisierung und Reglern
