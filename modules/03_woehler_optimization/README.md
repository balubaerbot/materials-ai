# Modul 3: Duell der Optimierer — Wöhlerlinie unter Data Scarcity

**Lernziel:** Verstehen, warum Gradient Descent in lokalen Minima stecken
bleiben kann und wie Simulated Annealing (aus der Metallurgie abgeleitet —
passend zum Thema!) solche Fallen überspringen kann.

## Szenario

Parameter-Identifikation einer Schweißnaht-Wöhlerlinie (Basquin-Gleichung
`N · (Δσ)^k = C`, logarithmiert `log(N) = log(C) + k · log(Δσ)`) aus nur
5–50 stark verrauschten Messpunkten — typisch für reale Schweißnaht-
Ermüdungsversuche (geometrische Toleranzen, Eigenspannungen).

**Wichtiger Modellierungs-Hinweis:** Die reine Regressions-Fehlerfläche ist
konvex (kein echtes lokale-Minima-Problem). Um das Duell "GD bleibt stecken
vs. SA entkommt" trotzdem sichtbar zu machen, überlagern wir eine
künstliche, periodische Wellblech-Struktur auf die Fehlerfläche (nur entlang
der k-Achse, empirisch kalibriert). Das ist ein gängiger didaktischer
Kunstgriff, um Optimierer-Verhalten auf einer echten, mehrgipfligen
Landschaft zu demonstrieren.

**Zweiter Hinweis:** k und log C sind in dieser Regression stark korreliert
(schmales, gebogenes Tal statt rundem Krater). Beide Optimierer lassen
log C deshalb über die geschlossene Profile-Lösung an das jeweilige k
"andocken" (`optimizers.logc_profile`), statt es unabhängig zu suchen —
sonst würden reine Zufallsschritte fast immer seitlich neben das Tal fallen.

Simulated Annealing berichtet als Endergebnis das **beste je gefundene**
(k, log C) über den gesamten Lauf — nicht die letzte (evtl. noch "heiße")
Position — wie in der Praxis üblich.

## Starten

```bash
docker compose up -d --build module-03-woehler-optimization
```

Dann im Browser (nur lokal/Tunnel, siehe Root-README): `http://localhost:8503`

## Dateien

- `data_generator.py` — synthetische Wöhlerlinie mit starkem Rauschen
- `optimizers.py` — Gradient Descent (PyTorch/Adam) + Simulated Annealing (NumPy) + gemeinsame Fehlerfunktion mit Soft-Constraint und künstlicher Ripple-Struktur
- `app.py` — Streamlit-Dashboard: 3-Panel-Live-Duell (Wöhler-Fit, Loss-Landschaft mit Pfaden, Loss-über-Zeit)
