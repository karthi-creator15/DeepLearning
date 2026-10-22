import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_breast_cancer()
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val = scaler.transform(X_val_raw).astype(np.float32)

class CsvDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_loader = DataLoader(CsvDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(CsvDataset(X_val, y_val), batch_size=32, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layers(x).squeeze(1)

def evaluate(model, loader):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(DEVICE))
            predicted.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int))
            actual.extend(labels.numpy().astype(int))
    return accuracy_score(actual, predicted), np.array(actual), np.array(predicted)

model = MLP(X_train.shape[1]).to(DEVICE)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_state = copy.deepcopy(model.state_dict())
best_accuracy, best_epoch, patience, waiting = 0.0, 0, 7, 0
loss_history = []

for epoch in range(1, 41):
    model.train()
    total_loss, count = 0.0, 0
    for features, labels in train_loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    loss_history.append(total_loss / count)
    val_accuracy, _, _ = evaluate(model, val_loader)
    print(f"Epoch {epoch:02d}: loss={loss_history[-1]:.4f}, validation accuracy={val_accuracy:.4f}")
    if val_accuracy > best_accuracy:
        best_accuracy, best_epoch = val_accuracy, epoch
        best_state = copy.deepcopy(model.state_dict())
        waiting = 0
    else:
        waiting += 1
    if waiting >= patience:
        print("Early stopping.")
        break

model.load_state_dict(best_state)
torch.save(model.state_dict(), "mlp_state_dict.pt")
torch.save({"mean": scaler.mean_, "scale": scaler.scale_}, "scaler_stats.pt")
reloaded = MLP(X_train.shape[1]).to(DEVICE)
reloaded.load_state_dict(torch.load("mlp_state_dict.pt", map_location=DEVICE, weights_only=True))
reloaded.eval()
original_accuracy, _, _ = evaluate(model, val_loader)
reloaded_accuracy, _, _ = evaluate(reloaded, val_loader)
print(f"Best validation accuracy: {best_accuracy:.4f}, epoch {best_epoch}")
print(f"Reloaded accuracy: {reloaded_accuracy:.4f}; identical: {abs(original_accuracy-reloaded_accuracy) < 1e-12}")

baseline = Pipeline([
    ("scale", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=2000, random_state=SEED))
])
baseline.fit(X_train_raw, y_train)
baseline_pred = baseline.predict(X_val_raw)
print("Logistic regression accuracy:", round(accuracy_score(y_val, baseline_pred), 4))
print("Logistic regression macro F1:", round(f1_score(y_val, baseline_pred, average="macro"), 4))

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("MLP training loss")
plt.tight_layout()
plt.savefig("training_loss.png")
plt.close()
