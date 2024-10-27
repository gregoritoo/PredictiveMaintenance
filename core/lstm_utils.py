import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def generate_triplets(x, y):
    triplets = []
    for i in range(len(y)):
        anchor = x[i]
        all_pos = [j for j in range(len(y)) if y[j] == y[i] and j != i]
        positive_idx = (
            torch.tensor(random.choice(all_pos), dtype=torch.long)
            if len(all_pos) > 0
            else torch.tensor(i, dtype=torch.long)
        )
        negative_idx = torch.tensor(
            random.choice([j for j in range(len(y)) if y[j] != y[i]]), dtype=torch.long
        )
        positive = x[positive_idx, :, :]
        negative = x[negative_idx, :, :]
        triplets.append([anchor, positive, negative])
    return triplets


def triplet_loss(anchor, positive, negative, margin=0.5):
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


class MultivariateRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout=0.1):
        super(MultivariateRNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_dim),
        )

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h_0, c_0)

        lstm_out, _ = self.lstm(x, hidden)

        last_hidden = lstm_out[:, -1, :]

        return last_hidden

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x)
        return output


def train_contrastive(model, optimizer, triplets, margin=0.5, epochs=10, device="mps"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm.tqdm(triplets):

            anchor, positive, negative = batch

            anchor_emb = model(anchor.unsqueeze(0).to(device))
            positive_emb = model(positive.unsqueeze(0).to(device))
            negative_emb = model(negative.unsqueeze(0).to(device))
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb, margin=margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(triplets)}")
    return model
