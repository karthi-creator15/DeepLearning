import numpy as np
import matplotlib.pyplot as plt
import torch

np.random.seed(1)
torch.manual_seed(1)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = np.asarray(x, dtype=float)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def mse(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred), 1e-9, 1 - 1e-9)
    y_true = np.asarray(y_true)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

def train_xor(lr=0.5, epochs=5000, seed=1):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, 4))
    b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 1, (4, 1))
    b2 = np.zeros((1, 1))
    losses = []
    for _ in range(epochs):
        z1 = X @ W1 + b1
        a1 = relu(z1)
        a2 = sigmoid(a1 @ W2 + b2)
        loss = cross_entropy(y, a2)
        losses.append(float(loss))
        dz2 = (a2 - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ W2.T) * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        if not np.isfinite(loss) or loss > 1e6:
            break
    predictions = sigmoid(relu(X @ W1 + b1) @ W2 + b2).ravel()
    return predictions, losses

print("softmax([2.0, 1.0, 0.1]) =", np.round(softmax([2.0, 1.0, 0.1]), 3))
predictions, losses = train_xor()
print("XOR probabilities:", np.round(predictions, 3))
print("XOR classes:", (predictions >= 0.5).astype(int))

plt.figure()
plt.plot(losses)
plt.title("XOR training loss (learning rate 0.5)")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.tight_layout()
plt.savefig("xor_loss.png")
plt.close()

for lr in [0.01, 5.0]:
    _, curve = train_xor(lr=lr)
    curve = [v for v in curve if np.isfinite(v)]
    print(f"Learning rate {lr}: epochs={len(curve)}, final loss={curve[-1] if curve else 'non-finite'}")
    plt.figure()
    plt.plot(curve)
    plt.title(f"XOR loss (learning rate {lr})")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.tight_layout()
    plt.savefig(f"xor_loss_lr_{str(lr).replace('.', '_')}.png")
    plt.close()

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
loss = (w * 2 + b - 10) ** 2
loss.backward()
print("Hand gradients: dw = -32, db = -16")
print("Autograd: w.grad =", w.grad.item(), "b.grad =", b.grad.item())
print("Dropout: randomly disables activations during training to reduce overfitting.")
print("BatchNorm: normalises layer activations within a mini-batch to stabilise training.")
print("Early stopping: stops when validation performance stops improving.")
