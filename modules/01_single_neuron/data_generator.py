"""
data_generator.py — Synthetische Werkstoffkennlinie (Spannung-Dehnung-Diagramm)

Simuliert das Verhalten von Baustahl:
  1. Elastischer Bereich (Hooke'sches Gesetz):   sigma = E * epsilon
  2. Fließbereich mit plastischer Verfestigung:  sigma = sigma_y + H * (epsilon - epsilon_y)^p

Damit erzeugen wir eine Kurve, die aus einem linearen und einem nichtlinearen
Abschnitt besteht — ideal, um später zu zeigen, wo ein einzelnes Neuron
(= lineare Regression) an seine Grenzen stößt.
"""

import numpy as np


def generate_steel_curve(
    n_points: int = 200,
    E: float = 210_000.0,       # Elastizitätsmodul [MPa] (typisch für Stahl)
    yield_strain: float = 0.002,  # Dehnung an der Streckgrenze
    yield_stress: float | None = None,  # optional override, sonst E * yield_strain
    hardening_modulus: float = 2_000.0,  # Verfestigungssteigung im plastischen Bereich [MPa]
    hardening_exponent: float = 0.5,     # <1 => abflachende Verfestigung (typisch)
    max_strain: float = 0.05,   # maximale Dehnung des Diagramms
    noise_std: float = 15.0,    # Rauschen in MPa (Messungenauigkeit)
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Erzeugt (strain, stress) Arrays einer synthetischen Werkstoffkennlinie.

    Returns
    -------
    strain : np.ndarray, shape (n_points,)
    stress : np.ndarray, shape (n_points,)
    """
    rng = np.random.default_rng(seed)

    if yield_stress is None:
        yield_stress = E * yield_strain

    strain = np.linspace(0.0, max_strain, n_points)
    stress = np.empty_like(strain)

    elastic_mask = strain <= yield_strain
    plastic_mask = ~elastic_mask

    # Elastischer Bereich: linear
    stress[elastic_mask] = E * strain[elastic_mask]

    # Plastischer Bereich: Verfestigung mit abflachendem Exponenten
    delta = strain[plastic_mask] - yield_strain
    stress[plastic_mask] = yield_stress + hardening_modulus * (delta ** hardening_exponent)

    # Messrauschen
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_points)
    stress = stress + noise

    return strain, stress


if __name__ == "__main__":
    # Kurzer Selbsttest
    eps, sig = generate_steel_curve()
    print(f"Dehnung: {eps.min():.4f} .. {eps.max():.4f}")
    print(f"Spannung: {sig.min():.1f} .. {sig.max():.1f} MPa")
