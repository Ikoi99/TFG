# vqc_qiskit.py
# vqc_qiskit.py
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize


# Data (same idea as above, 2x2)
digits = load_digits()
mask = (digits.target == 0) | (digits.target == 1)
X = digits.images[mask]
y = digits.target[mask]
X_ds = X.reshape(-1, 2, 4, 2, 4).mean(axis=(2,4))
X_flat = X_ds.reshape(len(X_ds), -1)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_flat)
y_bin = (y == 1).astype(float)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)

n_qubits = 4
n_layers = 2
theta = ParameterVector('t', n_layers * n_qubits * 2)

def build_circuit(x, thetas):
    qc = QuantumCircuit(n_qubits)
    # encoding
    for i in range(n_qubits):
        qc.ry(float(x[i]), i)
    # variational layers
    idx = 0
    for l in range(n_layers):
        for q in range(n_qubits):
            qc.ry(thetas[idx], q); idx += 1
            qc.rz(thetas[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q+1)
    qc.measure_all()
    return qc

backend = Aer.get_backend('aer_simulator')
shots = 1024

def circuit_expectation(x, params):
    qc = build_circuit(x, params)
    qc = qc.bind_parameters({theta[i]: params[i] for i in range(len(params))})
    # use Aer simulator
    qc.save_counts()
    job = execute(qc, backend=backend, shots=shots)
    counts = job.result().get_counts()
    # majority of measured bitstrings: we interpret first qubit bit (q0) as label
    # Qiskit returns bitstrings with qubit ordering; take bit at position -1 (q0)
    # Compute prob of q0 = 0
    total = sum(counts.values())
    p_q0_0 = sum(v for k,v in counts.items() if k[-1] == '0') / total
    return p_q0_0

# cost over dataset
def cost_fn(params):
    preds = np.array([circuit_expectation(x, params) for x in X_train[:20]])  # small batch to speed
    return np.mean((preds - y_train[:20])**2)

init = np.random.normal(0, 0.1, size=(n_layers * n_qubits * 2))
res = minimize(cost_fn, init, method='COBYLA', options={'maxiter': 80})
params_opt = res.x
# evaluate on test (small sample)
test_preds = np.array([circuit_expectation(x, params_opt) for x in X_test[:50]])
test_labels = (test_preds > 0.5).astype(int)
print("Test acc (sample):", (test_labels == y_test[:50]).mean())
