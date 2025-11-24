# vqc_pennylane.py
import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits

# --- Data: example using sklearn digits, filter two classes and downsample to 2x2 ---
digits = load_digits()
# Use digits 0 and 1 as example
mask = (digits.target == 0) | (digits.target == 1)
X = digits.images[mask]
y = digits.target[mask]
# downsample 8x8 -> 2x2 by average pooling (simple)
X_ds = X.reshape(-1, 2, 4, 2, 4).mean(axis=(2,4))  # from 8x8 -> 2x2
X_flat = X_ds.reshape(len(X_ds), -1)  # shape (N,4)
# Normalize to [0, pi]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_flat)
y_bin = (y == 1).astype(float)  # labels 0 or 1

# train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)

n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits, shots=1000)

def angle_encode(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

def variational_layer(params):
    # params shape: (n_layers, n_qubits, 2)
    for i in range(n_qubits):
        qml.RY(params[i,0], wires=i)
        qml.RZ(params[i,1], wires=i)
    # entangle linearly
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev, interface="autograd")
def circuit(x, flat_params):
    # reshape flat params to (n_layers, n_qubits, 2)
    params = flat_params.reshape((n_layers, n_qubits, 2))
    angle_encode(x)
    for l in range(n_layers):
        variational_layer(params[l])
    # measure expectation on first qubit
    return qml.expval(qml.PauliZ(0))

def sq_error(pred, target):
    # map expval (-1..1) -> prob (0..1)
    prob = (1 - pred) / 2
    return (prob - target) ** 2

# initialize parameters
np.random.seed(0)
init_params = np.random.normal(0, 0.1, size=(n_layers * n_qubits * 2))
opt = qml.optimize.AdamOptimizer(stepsize=0.1)

params = init_params.copy()
epochs = 60
batch_size = 8

for epoch in range(epochs):
    # simple mini-batch SGD
    idx = np.random.permutation(len(X_train))
    for start in range(0, len(X_train), batch_size):
        batch_idx = idx[start:start+batch_size]
        Xb = X_train[batch_idx]
        yb = y_train[batch_idx]
        def cost(p):
            preds = [circuit(x, p) for x in Xb]
            return np.mean([sq_error(pv, tv) for pv, tv in zip(preds, yb)])
        params = opt.step(cost, params)
    # train loss
    if epoch % 10 == 0 or epoch == epochs-1:
        train_preds = np.array([ (1 - circuit(x, params)) / 2 for x in X_train ])
        train_loss = np.mean((train_preds - y_train)**2)
        print(f"Epoch {epoch:3d}  train_mse={train_loss:.4f}")

# Evaluate on test
test_preds = np.array([ (1 - circuit(x, params)) / 2 for x in X_test ])
test_labels = (test_preds > 0.5).astype(int)
accuracy = (test_labels == y_test).mean()
print("Test accuracy:", accuracy)
