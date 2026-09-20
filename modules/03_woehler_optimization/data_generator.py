"""
data_generator.py — Synthetische Schweißnaht-Wöhlerlinie (S-N-Kurve)

Basquin-Gleichung:  N * (Δσ)^k = C
Logarithmiert:       log(N) = log(C) + k * log(Δσ)

Hinweis zur Vorzeichenkonvention: Wir definieren k hier direkt als die
**Steigung der Gerade** im doppelt-logarithmischen log(N)-vs-log(Δσ)-Diagramm.
Physikalisch ist diese Steigung negativ (mehr Spannung -> weniger Lastwechsel
bis Bruch), typischerweise um k ≈ -3 für ungekerbte Normalspannungs-Details
nach Eurocode 3. Das deckt sich mit der Vorgabe "k muss negativ sein, Soft-
Constraint um den Wert 3".

Wöhlerlinien für Schweißnähte streuen im Realversuch stark (geometrische
Toleranzen der Naht, Eigenspannungen aus dem Schweißprozess, Einschlüsse) —
das bilden wir über Gaußsches Rauschen auf log(N) ab.
"""

import numpy as np

TRUE_K = -3.0          # wahre Steigung (Ziel des Soft-Constraints)
TRUE_LOG_C = 17.0      # wahre Festigkeitskonstante (natürlicher Logarithmus)


def generate_woehler_data(
    n_points: int = 10,
    true_k: float = TRUE_K,
    true_log_c: float = TRUE_LOG_C,
    delta_sigma_range: tuple[float, float] = (80.0, 400.0),  # MPa, typ. Spannungsschwingbreite
    noise_std: float = 0.25,   # Streuung auf log(N), typ. 0.2-0.3 für Schweißnähte
    seed: int | None = 42,
):
    """
    Erzeugt (delta_sigma, N) Messpunkte einer synthetischen Wöhlerlinie.

    Returns
    -------
    delta_sigma : np.ndarray, shape (n_points,) — Spannungsschwingbreite [MPa]
    N : np.ndarray, shape (n_points,) — Lastwechsel bis Bruch
    """
    rng = np.random.default_rng(seed)

    # log-uniforme Stichprobe über den Spannungsbereich (typisch für Versuchsplanung)
    log_ds = rng.uniform(np.log(delta_sigma_range[0]), np.log(delta_sigma_range[1]), n_points)
    log_ds.sort()
    delta_sigma = np.exp(log_ds)

    log_N_true = true_log_c + true_k * log_ds
    noise = rng.normal(0.0, noise_std, n_points)
    log_N = log_N_true + noise

    N = np.exp(log_N)
    return delta_sigma, N


if __name__ == "__main__":
    ds, N = generate_woehler_data()
    print("Delta_sigma:", ds)
    print("N:", N)
