"""
app.py — Streamlit-Dashboard für Modul 3: Duell der Optimierer

Gradient Descent (rot) vs. Simulated Annealing (blau) bei der Parameter-
Identifikation einer Schweißnaht-Wöhlerlinie aus extrem wenigen, stark
verrauschten Messpunkten — auf einer künstlich aufgerauten Fehler-Landschaft
mit echten lokalen Minima.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from data_generator import generate_woehler_data, TRUE_K, TRUE_LOG_C
import optimizers as opt

st.set_page_config(page_title="Modul 3: Duell der Optimierer", layout="wide")

st.title("⚔️ Modul 3: Duell der Optimierer — Wöhlerlinie unter Data Scarcity")
st.markdown(
    """
Wir bestimmen die Parameter einer Schweißnaht-Wöhlerlinie (Basquin:
`log(N) = log(C) + k · log(Δσ)`) aus nur wenigen, stark verrauschten
Messpunkten. Die Fehlerfläche ist bewusst mit einer künstlichen
**Wellblech-Struktur** überlagert — reine Regression wäre konvex und hätte
keine lokalen Minima, aber genau die wollen wir hier sichtbar machen.

🔴 **Gradient Descent** (PyTorch/Adam) — folgt nur dem lokalen Gefälle und
bleibt typischerweise in der ersten Rippel-Mulde hängen.

🔵 **Simulated Annealing** — springt anfangs "heiß" wild umher (akzeptiert
auch Verschlechterungen) und kann Rippel überspringen, um das globale
Minimum zu finden.
"""
)

# ---------------------------------------------------------------------------
# Sidebar: Regler
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Messdaten")
n_points = st.sidebar.slider("Anzahl Messpunkte", 5, 50, 10, step=1)
noise_std = st.sidebar.slider("Messrauschen (auf log N)", 0.0, 0.6, 0.25, step=0.05)

st.sidebar.header("🔵 Simulated Annealing")
t_start = st.sidebar.slider("Starttemperatur", 0.5, 20.0, 8.0, step=0.5)

st.sidebar.header("🎯 Trainings-Parameter")
iterations = st.sidebar.slider("Schritte / Iterationen", 50, 1000, 300, step=50)
learning_rate = st.sidebar.select_slider(
    "Lernrate (Gradient Descent)", options=[0.005, 0.01, 0.03, 0.05, 0.1], value=0.03
)
animate = st.sidebar.checkbox("Live-Animation", value=True)

start = st.sidebar.button("⚔️ Duell starten", use_container_width=True)

INIT_K = -0.5  # gemeinsamer, naiver Startwert für beide Optimierer (fairer Vergleich)

# ---------------------------------------------------------------------------
# Daten generieren
# ---------------------------------------------------------------------------
delta_sigma, N = generate_woehler_data(n_points=n_points, noise_std=noise_std, seed=42)
log_ds = np.log(delta_sigma)
log_N = np.log(N)

# Statische Loss-Landschaft (einmal berechnet, unabhängig vom Training)
k_lin = np.linspace(-8.0, 1.0, 150)
logC_lin = np.linspace(-5.0, 35.0, 150)
K_grid, LC_grid = np.meshgrid(k_lin, logC_lin)
Z_grid = opt.loss_grid(K_grid, LC_grid, log_ds, log_N)
Z_plot = np.clip(Z_grid, None, 8.0)  # Ausreißer kappen, sonst dominiert die Farbskala

plot_placeholder = st.empty()
metrics_placeholder = st.empty()
result_placeholder = st.empty()


def draw(gd_point=None, sa_point=None, gd_hist=None, sa_hist=None, sa_best=None, step_label=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # --- Panel 1: Wöhler-Kurve (log-log) ---
    ax = axes[0]
    ax.loglog(delta_sigma, N, "o", color="steelblue", alpha=0.6, label="Messdaten")
    ds_line = np.linspace(delta_sigma.min(), delta_sigma.max(), 100)
    if gd_point is not None:
        n_line = np.exp(gd_point["logC"] + gd_point["k"] * np.log(ds_line))
        ax.loglog(ds_line, n_line, color="crimson", linewidth=2, label=f"GD (k={gd_point['k']:.2f})")
    if sa_point is not None:
        n_line = np.exp(sa_point["logC"] + sa_point["k"] * np.log(ds_line))
        ax.loglog(ds_line, n_line, color="royalblue", linewidth=1.5, alpha=0.5, linestyle=":",
                   label=f"SA aktuell (k={sa_point['k']:.2f})")
    if sa_best is not None:
        n_line = np.exp(sa_best["logC"] + sa_best["k"] * np.log(ds_line))
        ax.loglog(ds_line, n_line, color="royalblue", linewidth=2,
                   label=f"SA bestes (k={sa_best['k']:.2f})")
    ax.set_xlabel("Δσ [MPa] (log)")
    ax.set_ylabel("N [Lastwechsel] (log)")
    ax.set_title("Wöhlerlinie: Daten vs. Fits")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # --- Panel 2: Loss-Landschaft mit Pfaden ---
    ax2 = axes[1]
    cs = ax2.contourf(K_grid, LC_grid, Z_plot, levels=30, cmap="viridis")
    if gd_hist:
        ax2.plot([h["k"] for h in gd_hist], [h["logC"] for h in gd_hist], color="crimson", linewidth=1.5, alpha=0.8)
        ax2.plot(gd_hist[-1]["k"], gd_hist[-1]["logC"], "o", color="crimson", markersize=8)
    if sa_hist:
        ax2.plot([h["k"] for h in sa_hist], [h["logC"] for h in sa_hist], color="royalblue", linewidth=0.8, alpha=0.5)
        ax2.plot(sa_hist[-1]["k"], sa_hist[-1]["logC"], "o", color="royalblue", markersize=6, alpha=0.6)
    if sa_best is not None:
        ax2.plot(sa_best["k"], sa_best["logC"], "*", color="gold", markersize=18,
                  markeredgecolor="black", markeredgewidth=0.8, label="SA bestes")
    ax2.axvline(TRUE_K, color="white", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("k")
    ax2.set_ylabel("log C")
    ax2.set_title(f"Fehler-Landschaft {step_label}")

    # --- Panel 3: Loss über Zeit ---
    ax3 = axes[2]
    if gd_hist:
        ax3.plot([h["step"] for h in gd_hist], [h["loss"] for h in gd_hist], color="crimson", label="GD")
    if sa_hist:
        ax3.plot([h["step"] for h in sa_hist], [h["loss"] for h in sa_hist], color="royalblue", label="SA (aktuell, springt weiter)", alpha=0.6)
    if sa_best is not None:
        ax3.axhline(sa_best["loss"], color="gold", linestyle="--", linewidth=1, label="SA bestes")
    ax3.set_yscale("log")
    ax3.set_xlabel("Schritt")
    ax3.set_ylabel("Loss (log)")
    ax3.set_title("Trainingsverlauf")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    plot_placeholder.pyplot(fig)
    plt.close(fig)


draw()

if start:
    opt.LAMBDA_CONSTRAINT = 0.1
    opt.RIPPLE_AMP = 0.15
    opt.RIPPLE_PERIOD_K = 0.5

    cooling_rate = 0.01 ** (1.0 / iterations)  # T sinkt bis Ende auf ~1% von t_start

    gd_full = opt.run_gradient_descent(log_ds, log_N, INIT_K, steps=iterations, learning_rate=learning_rate, log_every=1)
    sa_full = opt.run_simulated_annealing(
        log_ds, log_N, INIT_K, iterations=iterations, t_start=t_start,
        cooling_rate=cooling_rate, seed=7,
    )

    n_frames = min(80, iterations)
    frame_idx = np.linspace(1, len(gd_full) - 1, n_frames).astype(int)

    if animate:
        for i in frame_idx:
            gd_slice = gd_full[: i + 1]
            sa_slice = sa_full[: i + 1]
            sa_best_so_far = min(sa_slice, key=lambda h: h["loss"])
            draw(
                gd_point=gd_slice[-1], sa_point=sa_slice[-1],
                gd_hist=gd_slice, sa_hist=sa_slice, sa_best=sa_best_so_far,
                step_label=f"(Schritt {gd_slice[-1]['step']}/{iterations})",
            )
            metrics_placeholder.info(
                f"Schritt {gd_slice[-1]['step']}/{iterations} — "
                f"🔴 GD: k={gd_slice[-1]['k']:.2f}, Loss={gd_slice[-1]['loss']:.4f}  |  "
                f"🔵 SA: k={sa_slice[-1]['k']:.2f}, Loss={sa_slice[-1]['loss']:.4f}"
            )
            time.sleep(0.03)

    # SA-Ergebnis: bestes je gefundenes (k, logC), nicht die letzte (weiter "heiße") Position
    sa_best = min(sa_full, key=lambda h: h["loss"])
    gd_final = gd_full[-1]

    if not animate:
        draw(
            gd_point=gd_full[-1], sa_point=sa_full[-1],
            gd_hist=gd_full, sa_hist=sa_full, sa_best=sa_best,
            step_label=f"(Schritt {iterations}/{iterations})",
        )
    else:
        # letzter Frame nochmal mit sa_best-Stern, damit er auch bei fertiger Animation sichtbar bleibt
        draw(
            gd_point=gd_full[-1], sa_point=sa_full[-1],
            gd_hist=gd_full, sa_hist=sa_full, sa_best=sa_best,
            step_label=f"(Schritt {iterations}/{iterations})",
        )

    with result_placeholder.container():
        st.subheader("📊 Ergebnis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Wahre Werte**")
            st.metric("k", f"{TRUE_K:.2f}")
            st.metric("log C", f"{TRUE_LOG_C:.2f}")
        with col2:
            st.markdown("**🔴 Gradient Descent**")
            st.metric("k", f"{gd_final['k']:.2f}", delta=f"{gd_final['k'] - TRUE_K:+.2f}")
            st.metric("log C", f"{gd_final['logC']:.2f}", delta=f"{gd_final['logC'] - TRUE_LOG_C:+.2f}")
        with col3:
            st.markdown("**🔵 Simulated Annealing** (bestes Ergebnis)")
            st.metric("k", f"{sa_best['k']:.2f}", delta=f"{sa_best['k'] - TRUE_K:+.2f}")
            st.metric("log C", f"{sa_best['logC']:.2f}", delta=f"{sa_best['logC'] - TRUE_LOG_C:+.2f}")

        if abs(sa_best["k"] - TRUE_K) < abs(gd_final["k"] - TRUE_K):
            st.success(
                "✅ Wie erwartet: Simulated Annealing kam näher an das wahre k heran — "
                "Gradient Descent blieb in einer lokalen Rippel-Mulde nahe dem Startwert stecken."
            )
        else:
            st.warning(
                "Diesmal lag Gradient Descent näher am wahren Wert — probier andere Messrauschen-/"
                "Iterations-Einstellungen, um das Ripple-Verhalten stärker hervortreten zu lassen."
            )
