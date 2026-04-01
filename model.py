"""
model.py  —  Action Classifier (LSTM)
TODO: Train this on your collected keypoint sequences, then
      import it in main.py to replace PlaceholderClassifier.
"""
import torch
import torch.nn as nn


class ActionLSTM(nn.Module):
    """
    Input  : (batch, seq_len, input_size)   e.g. (B, 30, 132)
    Output : (batch, num_classes)
    """
    def __init__(self, input_size=132, hidden_size=128,
                 num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])   # last time step


if __name__ == "__main__":
    model = ActionLSTM()
    dummy = torch.randn(4, 30, 132)
    print("Output shape:", model(dummy).shape)  # (4, 6)
