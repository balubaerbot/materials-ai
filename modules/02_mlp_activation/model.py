"""
model.py — Mehrschichtiges neuronales Netz (MLP) mit Aktivierungsfunktion.

Modul 1 hat gezeigt: ein einzelnes Neuron kann nur eine Gerade lernen.
Hier stapeln wir mehrere Neuronen in versteckten Schichten und schalten
zwischen ihnen eine nichtlineare Aktivierungsfunktion (ReLU).

Warum das reicht: Ohne Aktivierungsfunktion wäre jede Verkettung von linearen
Schichten wieder nur eine einzige lineare Abbildung (Matrixmultiplikation
bleibt linear). Erst die Nichtlinearität zwischen den Schichten erlaubt es
dem Netz, Knicke und gekrümmte Verläufe abzubilden — wie den Übergang von
elastischem zu plastischem Verhalten an der Streckgrenze.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Mehrschichtiges Netz: Eingang -> [Linear -> ReLU] * n_hidden_layers -> Ausgang."""

    def __init__(self, hidden_size: int = 16, n_hidden_layers: int = 2):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(1, hidden_size), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    strain: torch.Tensor,
    stress: torch.Tensor,
    hidden_size: int = 16,
    n_hidden_layers: int = 2,
    epochs: int = 500,
    learning_rate: float = 0.01,
    log_every: int = 5,
):
    """
    Trainiert das MLP per Gradientenabstieg (Adam) auf Mean-Squared-Error.

    Normalisiert Eingabe/Ausgabe intern für numerische Stabilität.

    Returns
    -------
    model : trainiertes MLP
    history : list of dicts {epoch, loss, predict_fn}
        predict_fn(x_real: np.ndarray) -> np.ndarray gibt die Vorhersage in
        Original-Skala für diesen Trainingsstand zurück (für Plot-Snapshots).
    """
    x_mean, x_std = strain.mean(), strain.std() + 1e-8
    y_mean, y_std = stress.mean(), stress.std() + 1e-8

    x_norm = (strain - x_mean) / x_std
    y_norm = (stress - y_mean) / y_std

    model = MLP(hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = []

    def make_predict_fn(state_dict):
        snapshot = MLP(hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
        snapshot.load_state_dict(state_dict)
        snapshot.eval()

        def predict(x_real_np):
            with torch.no_grad():
                x_t = torch.tensor(x_real_np, dtype=torch.float32).unsqueeze(1)
                x_t_norm = (x_t - x_mean) / x_std
                y_pred_norm = snapshot(x_t_norm)
                y_pred = y_pred_norm * y_std + y_mean
            return y_pred.squeeze(1).numpy()

        return predict

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(x_norm)
        loss = loss_fn(pred, y_norm)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            state_copy = {k: v.clone() for k, v in model.state_dict().items()}
            history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "predict_fn": make_predict_fn(state_copy),
            })

    return model, history
