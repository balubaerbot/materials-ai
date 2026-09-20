# Modul 2: Mehrschichtiges Netz mit Aktivierungsfunktion

**Lernziel:** Verstehen, warum mehrere Neuronen + eine nichtlineare
Aktivierungsfunktion (ReLU) nötig sind, um gekrümmte/geknickte Verläufe
abzubilden — und warum reine Verkettung linearer Schichten ohne Aktivierung
nichts bringt (bleibt mathematisch linear).

## Szenario

Gleiche Werkstoffkennlinie wie Modul 1 (Stahl, elastisch + plastisch
verfestigend). Diesmal fittet ein **Multi-Layer Perceptron (MLP)**:

```
Eingang (Dehnung) -> Linear -> ReLU -> Linear -> ReLU -> ... -> Linear -> Ausgang (Spannung)
```

Im Vergleich zu Modul 1 solltest du sehen: der Knick an der Streckgrenze und
die abflachende Verfestigungskurve werden jetzt deutlich besser getroffen.

## Starten

```bash
docker compose up -d --build module-02-mlp-activation
```

Dann im Browser (nur lokal/Tunnel, siehe Root-README): `http://localhost:8502`

## Dateien

- `data_generator.py` — gleiche synthetische Werkstoffkennlinie wie Modul 1
- `model.py` — das MLP (PyTorch, konfigurierbare Hidden-Layer/Neuronen) + Trainingsloop
- `app.py` — Streamlit-Dashboard mit Architektur-Reglern und Live-Visualisierung
