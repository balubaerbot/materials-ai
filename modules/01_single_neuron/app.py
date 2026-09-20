"""
app.py — Streamlit-Dashboard für Modul 1: Das einfachste Neuron

Zeigt live, wie ein einzelnes Neuron (= lineare Regression) versucht,
eine Werkstoffkennlinie zu fitten, und wo es an seine Grenzen stößt.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from data_generator import generate_steel_curve
from model import SingleNeuron, train_model

st.set_page_config(page_title="Modul 1: Das einfachste Neuron", layout="wide")

st.title("🔩 Modul 1: Das einfachste Neuron")
st.markdown(
    """
Ein einzelnes künstliches Neuron ohne Aktivierungsfunktion ist mathematisch
nichts anderes als eine **lineare Regression**: `y = w·x + b`.

Hier fitten wir damit eine Spannungs-Dehnungs-Kurve von Stahl. Der elastische
(lineare) Bereich sollte gut passen — der plastische (nichtlineare)
Verfestigungsbereich wird das Neuron nur grob mitteln können.
"""
)

# ---------------------------------------------------------------------------
# Sidebar: Regler
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Werkstoff-Parameter")
n_points = st.sidebar.slider("Anzahl Messpunkte", 50, 500, 200, step=10)
noise_std = st.sidebar.slider("Messrauschen [MPa]", 0.0, 50.0, 15.0, step=1.0)
E = st.sidebar.slider("E-Modul [MPa]", 100_000, 250_000, 210_000, step=5_000)
hardening_modulus = st.sidebar.slider("Verfestigungsmodul H [MPa]", 500, 5_000, 2_000, step=100)

st.sidebar.header("🧠 Trainings-Parameter")
epochs = st.sidebar.slider("Epochen", 10, 1000, 200, step=10)
learning_rate = st.sidebar.select_slider(
    "Lernrate", options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2], value=0.05
)
animate = st.sidebar.checkbox("Live-Animation beim Training", value=True)

start = st.sidebar.button("🚀 Training starten", use_container_width=True)

# ---------------------------------------------------------------------------
# Daten generieren
# ---------------------------------------------------------------------------
strain, stress = generate_steel_curve(
    n_points=n_points,
    E=E,
    hardening_modulus=hardening_modulus,
    noise_std=noise_std,
)

x = torch.tensor(strain, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(stress, dtype=torch.float32).unsqueeze(1)

plot_placeholder = st.empty()
metrics_placeholder = st.empty()


def draw(strain, stress, w=None, b=None, epoch=None, loss=None, loss_history=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.scatter(strain, stress, s=10, alpha=0.5, label="Messdaten (synthetisch)", color="steelblue")
    if w is not None:
        x_line = np.linspace(strain.min(), strain.max(), 50)
        y_line = w * x_line + b
        ax.plot(x_line, y_line, color="crimson", linewidth=2, label=f"Neuron-Fit (Epoche {epoch})")
    ax.set_xlabel("Dehnung ε [-]")
    ax.set_ylabel("Spannung σ [MPa]")
    ax.set_title("Werkstoffkennlinie: Daten vs. Neuron-Fit")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    if loss_history:
        epochs_ = [h["epoch"] for h in loss_history]
        losses_ = [h["loss"] for h in loss_history]
        ax2.plot(epochs_, losses_, color="darkorange")
        ax2.set_yscale("log")
    ax2.set_xlabel("Epoche")
    ax2.set_ylabel("Loss (MSE, normalisiert)")
    ax2.set_title("Trainingsverlauf")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_placeholder.pyplot(fig)
    plt.close(fig)


# Initialer Plot ohne Fit
draw(strain, stress)

if start:
    model, history = train_model(x, y, epochs=epochs, learning_rate=learning_rate)

    if animate:
        shown_history = []
        for point in history:
            shown_history.append(point)
            draw(
                strain, stress,
                w=point["w"], b=point["b"],
                epoch=point["epoch"], loss=point["loss"],
                loss_history=shown_history,
            )
            metrics_placeholder.info(
                f"Epoche {point['epoch']}/{epochs} — Loss: {point['loss']:.5f} — "
                f"w={point['w']:.1f}, b={point['b']:.1f}"
            )
            time.sleep(0.03)
    else:
        final = history[-1]
        draw(
            strain, stress,
            w=final["w"], b=final["b"],
            epoch=final["epoch"], loss=final["loss"],
            loss_history=history,
        )

    final = history[-1]
    st.success(
        f"✅ Training abgeschlossen. Finales Neuron: "
        f"σ ≈ {final['w']:.1f} · ε + {final['b']:.1f}  (Loss: {final['loss']:.5f})"
    )
    st.markdown(
        """
**Beobachtung:** Das Neuron findet eine gute Kompromiss-Gerade, kann aber weder
den scharfen Knick an der Streckgrenze noch die abflachende Verfestigungskurve
abbilden — dafür braucht es mehr als ein Neuron. Das ist Thema von Modul 2.
"""
    )
