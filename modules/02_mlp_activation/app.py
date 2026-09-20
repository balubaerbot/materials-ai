"""
app.py — Streamlit-Dashboard für Modul 2: Mehrschichtiges Netz mit Aktivierungsfunktion

Zeigt live, wie ein MLP (mehrere Neuronen + ReLU-Aktivierung) die
Werkstoffkennlinie deutlich besser fittet als das Einzel-Neuron aus Modul 1 —
inklusive des Knicks an der Streckgrenze.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from data_generator import generate_steel_curve
from model import train_model

st.set_page_config(page_title="Modul 2: MLP mit Aktivierungsfunktion", layout="wide")

st.title("🧠 Modul 2: Mehrschichtiges Netz mit Aktivierungsfunktion")
st.markdown(
    """
In Modul 1 hat ein einzelnes Neuron (= lineare Regression) versucht, die
Werkstoffkennlinie zu fitten — und konnte den Knick an der Streckgrenze nicht
abbilden, egal wie lange trainiert wurde.

Hier stapeln wir mehrere Neuronen in versteckten Schichten und schalten
**ReLU** (Rectified Linear Unit) dazwischen: `ReLU(x) = max(0, x)`.
Diese Nichtlinearität zwischen den Schichten ist der Grund, warum ein
mehrschichtiges Netz gekrümmte/geknickte Verläufe abbilden kann — eine reine
Verkettung linearer Schichten *ohne* Aktivierung bliebe mathematisch immer
noch nur eine Gerade.
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

st.sidebar.header("🧠 Netz-Architektur")
hidden_size = st.sidebar.slider("Neuronen pro Hidden-Layer", 2, 64, 16, step=2)
n_hidden_layers = st.sidebar.slider("Anzahl Hidden-Layer", 1, 4, 2, step=1)

st.sidebar.header("🎯 Trainings-Parameter")
epochs = st.sidebar.slider("Epochen", 50, 2000, 500, step=50)
learning_rate = st.sidebar.select_slider(
    "Lernrate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01
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


def draw(strain, stress, predict_fn=None, epoch=None, loss_history=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.scatter(strain, stress, s=10, alpha=0.5, label="Messdaten (synthetisch)", color="steelblue")
    if predict_fn is not None:
        x_line = np.linspace(strain.min(), strain.max(), 200)
        y_line = predict_fn(x_line)
        ax.plot(x_line, y_line, color="crimson", linewidth=2, label=f"MLP-Fit (Epoche {epoch})")
    ax.set_xlabel("Dehnung ε [-]")
    ax.set_ylabel("Spannung σ [MPa]")
    ax.set_title("Werkstoffkennlinie: Daten vs. MLP-Fit")
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


draw(strain, stress)

if start:
    model, history = train_model(
        x, y,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    if animate:
        shown_history = []
        for point in history:
            shown_history.append(point)
            draw(
                strain, stress,
                predict_fn=point["predict_fn"],
                epoch=point["epoch"],
                loss_history=shown_history,
            )
            metrics_placeholder.info(
                f"Epoche {point['epoch']}/{epochs} — Loss: {point['loss']:.5f} — "
                f"Architektur: {n_hidden_layers} Hidden-Layer × {hidden_size} Neuronen"
            )
            time.sleep(0.02)
    else:
        final = history[-1]
        draw(
            strain, stress,
            predict_fn=final["predict_fn"],
            epoch=final["epoch"],
            loss_history=history,
        )

    final = history[-1]
    st.success(
        f"✅ Training abgeschlossen. Finaler Loss: {final['loss']:.5f} "
        f"(Architektur: {n_hidden_layers} Hidden-Layer × {hidden_size} Neuronen)"
    )
    st.markdown(
        """
**Beobachtung:** Im Gegensatz zu Modul 1 (Einzel-Neuron) kann das MLP jetzt
den Knick an der Streckgrenze und die abflachende Verfestigungskurve
deutlich besser abbilden. Probier aus, was passiert, wenn du die
Hidden-Layer/Neuronen stark reduzierst (z. B. 1 Layer × 2 Neuronen) —
das Netz nähert sich dann wieder dem Verhalten aus Modul 1 an.
"""
    )
