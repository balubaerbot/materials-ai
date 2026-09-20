"""
model.py — Das einfachste neuronale Netz: ein einzelnes Neuron.

Ein Neuron ohne versteckte Schichten und ohne Aktivierungsfunktion ist
mathematisch nichts anderes als eine lineare Regression:

    y = w * x + b

Genau das ist der Punkt dieses Moduls: bevor wir "richtige" neuronale Netze
bauen, sehen wir, was ein einzelnes lineares Neuron kann — und wo es
versagt (nämlich bei der nichtlinearen plastischen Verfestigung).
"""

import torch
import torch.nn as nn


class SingleNeuron(nn.Module):
    """Ein Neuron: eine gewichtete Summe (hier: 1 Eingang) + Bias."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_model(
    strain: torch.Tensor,
    stress: torch.Tensor,
    epochs: int = 200,
    learning_rate: float = 0.01,
    log_every: int = 5,
):
    """
    Trainiert das Einzel-Neuron per Gradientenabstieg (Adam) auf Mean-Squared-Error.

    Skaliert Eingabe/Ausgabe intern für numerische Stabilität (Spannungen liegen
    im Bereich von hunderten MPa, Dehnungen im Promillebereich).

    Yields (als Liste, nicht als Generator, damit Streamlit einfach iterieren kann):
        history: list of dicts {epoch, loss, w, b}
    """
    # Normalisierung für stabiles Training
    x_mean, x_std = strain.mean(), strain.std() + 1e-8
    y_mean, y_std = stress.mean(), stress.std() + 1e-8

    x_norm = (strain - x_mean) / x_std
    y_norm = (stress - y_mean) / y_std

    model = SingleNeuron()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(x_norm)
        loss = loss_fn(pred, y_norm)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            w_norm = model.linear.weight.item()
            b_norm = model.linear.bias.item()

            # Gewichte zurück in die Original-Skala umrechnen:
            # y_norm = w_norm * x_norm + b_norm
            # (y - y_mean)/y_std = w_norm * (x - x_mean)/x_std + b_norm
            # y = (w_norm * y_std / x_std) * x + (y_mean + b_norm*y_std - w_norm*y_std*x_mean/x_std)
            w_real = w_norm * y_std / x_std
            b_real = y_mean + b_norm * y_std - w_norm * y_std * x_mean / x_std

            history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "w": w_real.item() if hasattr(w_real, "item") else float(w_real),
                "b": b_real.item() if hasattr(b_real, "item") else float(b_real),
            })

    return model, history
