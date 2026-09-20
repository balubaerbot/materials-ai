"""
data_generator.py — Synthetische Werkstoffkennlinie (Spannung-Dehnung-Diagramm)

Identisch zu Modul 1 (bewusst dupliziert statt importiert — jedes Modul bleibt
isoliert und eigenständig lauffähig, siehe Architektur-Prinzip im Root-README).

Simuliert das Verhalten von Baustahl:
  1. Elastischer Bereich (Hooke'sches Gesetz):   sigma = E * epsilon
  2. Fließbereich mit plastischer Verfestigung:  sigma = sigma_y + H * (epsilon - epsilon_y)^p
"""

import numpy as np


def generate_steel_curve(
    n_points: int = 200,
    E: float = 210_000.0,
    yield_strain: float = 0.002,
    yield_stress: float | None = None,
    hardening_modulus: float = 2_000.0,
    hardening_exponent: float = 0.5,
    max_strain: float = 0.05,
    noise_std: float = 15.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt (strain, stress) Arrays einer synthetischen Werkstoffkennlinie."""
    rng = np.random.default_rng(seed)

    if yield_stress is None:
        yield_stress = E * yield_strain

    strain = np.linspace(0.0, max_strain, n_points)
    stress = np.empty_like(strain)

    elastic_mask = strain <= yield_strain
    plastic_mask = ~elastic_mask

    stress[elastic_mask] = E * strain[elastic_mask]

    delta = strain[plastic_mask] - yield_strain
    stress[plastic_mask] = yield_stress + hardening_modulus * (delta ** hardening_exponent)

    noise = rng.normal(loc=0.0, scale=noise_std, size=n_points)
    stress = stress + noise

    return strain, stress


if __name__ == "__main__":
    eps, sig = generate_steel_curve()
    print(f"Dehnung: {eps.min():.4f} .. {eps.max():.4f}")
    print(f"Spannung: {sig.min():.1f} .. {sig.max():.1f} MPa")
