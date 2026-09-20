"""
optimizers.py — Duell der Optimierer: Gradient Descent vs. Simulated Annealing

Fehlerfunktion (gemeinsam für beide Verfahren):
    Loss(k, logC) = MSE(Fit, Messdaten)
                  + Soft-Constraint: (k - k_ziel)^2   [zieht k Richtung -3]
                  + Hard-Constraint: relu(k)^2         [bestraft k >= 0 stark]
                  + Ripple-Term: künstliche, periodische Komponente

Der Ripple-Term ist eine bewusste didaktische Zutat: eine reine Regressions-
Fehlerfläche (quadratisch in k/logC) ist konvex und hat KEINE lokalen Minima —
Gradient Descent würde dort immer gewinnen. Um das Szenario "GD bleibt in
lokalen Minima stecken, SA entkommt" überhaupt zeigen zu können, überlagern
wir eine periodische Rippel-Struktur auf die Fehlerfläche. Das ist ein
gängiger Kunstgriff, um Optimierer-Verhalten auf einer echten, mehrgipfligen
Landschaft sichtbar zu machen (reale, extrem datenarme Wöhler-Regressionen
können durch Rauschen ähnlich unruhige Landschaften erzeugen — hier machen
wir es deterministisch und reproduzierbar sichtbar).
"""

import numpy as np
import torch

K_TARGET = -3.0            # Soft-Constraint-Ziel (Eurocode-Normalspannungssteigung)
LAMBDA_CONSTRAINT = 0.1    # Gewicht des Soft-Constraints (sanfter Nudge, MSE trägt die Hauptlast)
HARD_PENALTY_WEIGHT = 5.0  # Gewicht der Hard-Constraint-Strafe (k < 0)
RIPPLE_AMP = 0.15          # Amplitude der künstlichen Rippel-Struktur (nur entlang k-Achse) —
                           # empirisch kalibriert: genug, damit GD von einem naiven Startwert
                           # (-0.5) nicht mehr wegkommt, aber SA über viele Wellen hinweg
                           # trotzdem verlässlich das globale Minimum bei k≈-3 findet.
RIPPLE_PERIOD_K = 0.5      # Periode der Rippel entlang der k-Achse — Wellblech-Muster mit
                           # Vertiefungen exakt bei Vielfachen von 0.5 (trifft k=-3 exakt)


def loss_numpy(k, logC, log_delta_sigma, log_N):
    """
    Fehlerfunktion in NumPy — funktioniert für Skalare UND Grids (k, logC
    beliebiger Shape, log_delta_sigma/log_N als 1D-Array der Messpunkte).
    """
    k = np.asarray(k, dtype=np.float64)
    logC = np.asarray(logC, dtype=np.float64)

    pred = logC[..., None] + k[..., None] * log_delta_sigma
    mse = np.mean((pred - log_N) ** 2, axis=-1)

    constraint = LAMBDA_CONSTRAINT * (k - K_TARGET) ** 2
    hard_penalty = HARD_PENALTY_WEIGHT * np.maximum(k, 0.0) ** 2
    ripple = RIPPLE_AMP * (1 - np.cos(2 * np.pi * k / RIPPLE_PERIOD_K))

    return mse + constraint + hard_penalty + ripple


def loss_torch(k: torch.Tensor, logC: torch.Tensor, log_delta_sigma_t: torch.Tensor, log_N_t: torch.Tensor):
    """Identische Fehlerfunktion, differenzierbar in PyTorch (für Gradient Descent)."""
    pred = logC + k * log_delta_sigma_t
    mse = torch.mean((pred - log_N_t) ** 2)

    constraint = LAMBDA_CONSTRAINT * (k - K_TARGET) ** 2
    hard_penalty = HARD_PENALTY_WEIGHT * torch.clamp(k, min=0.0) ** 2
    ripple = RIPPLE_AMP * (1 - torch.cos(2 * np.pi * k / RIPPLE_PERIOD_K))

    return mse + constraint + hard_penalty + ripple


def loss_grid(k_grid: np.ndarray, logC_grid: np.ndarray, log_delta_sigma, log_N):
    """Bequemlichkeitsfunktion für die Loss-Landschaft (2D-Grid -> 2D-Loss-Werte)."""
    return loss_numpy(k_grid, logC_grid, log_delta_sigma, log_N)


def logc_profile(k, log_delta_sigma, log_N):
    """
    Optimales logC für ein festgehaltenes k (geschlossene Lösung der linearen
    Regression, "Profile/Concentrated Likelihood"). k und logC sind in dieser
    Aufgabe stark korreliert — sie bilden ein schmales, gebogenes Tal statt
    eines runden Kraters. Beide Optimierer lassen logC deshalb an dieses k
    "andocken", statt es unabhängig zu suchen — dadurch bewegen sich beide
    Verfahren immer entlang des Tal-Bodens, nie seitlich daneben.
    """
    return np.mean(log_N) - k * np.mean(log_delta_sigma)


def logc_profile_torch(k: torch.Tensor, log_delta_sigma_t: torch.Tensor, log_N_t: torch.Tensor):
    """PyTorch-Variante von logc_profile (autograd-fähig, für Gradient Descent)."""
    return torch.mean(log_N_t) - k * torch.mean(log_delta_sigma_t)


def run_gradient_descent(
    log_delta_sigma: np.ndarray,
    log_N: np.ndarray,
    init_k: float,
    steps: int = 300,
    learning_rate: float = 0.05,
    log_every: int = 1,
):
    """
    Standard Gradient Descent (Adam) — deterministisch, folgt dem lokalen
    Gradienten. Optimiert nur k; logC wird pro Schritt aus dem Profile
    (logc_profile_torch) mitgezogen, damit die Suche immer im Tal bleibt.
    """
    log_ds_t = torch.tensor(log_delta_sigma, dtype=torch.float64)
    log_N_t = torch.tensor(log_N, dtype=torch.float64)

    k = torch.tensor(init_k, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([k], lr=learning_rate)

    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logC = logc_profile_torch(k, log_ds_t, log_N_t)
        loss = loss_torch(k, logC, log_ds_t, log_N_t)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1 or step == steps:
            history.append({
                "step": step,
                "k": k.item(),
                "logC": logc_profile(k.item(), log_delta_sigma, log_N),
                "loss": loss.item(),
            })

    return history


def run_simulated_annealing(
    log_delta_sigma: np.ndarray,
    log_N: np.ndarray,
    init_k: float,
    iterations: int = 300,
    t_start: float = 5.0,
    cooling_rate: float = 0.985,
    step_scale_k: float = 0.6,
    seed: int | None = None,
):
    """
    Simulated Annealing: springt bei hoher Temperatur wild ("heiß") und
    akzeptiert dabei auch Verschlechterungen (Metropolis-Kriterium) — das
    erlaubt, aus lokalen Rippel-Minima wieder herauszuspringen. Mit
    sinkender Temperatur wird das Verfahren zunehmend "gierig" und
    konvergiert gegen ein (im Idealfall globales) Minimum.

    Wie bei Gradient Descent wird nur k per Zufallsschritt vorgeschlagen;
    logC dockt über logc_profile immer optimal an das aktuelle k an, damit
    die Suche im schmalen, korrelierten Tal bleibt statt seitlich abzudriften.
    """
    rng = np.random.default_rng(seed)

    k = float(init_k)
    logC = logc_profile(k, log_delta_sigma, log_N)
    current_loss = float(loss_numpy(k, logC, log_delta_sigma, log_N))

    T = t_start
    history = [{"step": 0, "k": k, "logC": logC, "loss": current_loss, "T": T, "accepted": True}]

    for step in range(1, iterations + 1):
        temp_factor = T / t_start
        cand_k = k + rng.normal(0.0, step_scale_k) * temp_factor
        cand_logC = logc_profile(cand_k, log_delta_sigma, log_N)
        cand_loss = float(loss_numpy(cand_k, cand_logC, log_delta_sigma, log_N))

        delta = cand_loss - current_loss
        accept = delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-6))

        if accept:
            k, current_loss = cand_k, cand_loss
            logC = cand_logC

        T *= cooling_rate
        history.append({
            "step": step, "k": k, "logC": logC, "loss": current_loss,
            "T": T, "accepted": bool(accept),
        })

    return history
