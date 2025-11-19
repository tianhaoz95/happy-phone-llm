import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

# Input
input_tensor = np.array([[0.5, 1.0]], dtype=np.float32)

# Weights
w1 = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8]
], dtype=np.float32)

w3 = np.array([
    [0.8, 0.7, 0.6, 0.5],
    [0.4, 0.3, 0.2, 0.1]
], dtype=np.float32)

w2 = np.array([
    [0.9, 1.0],
    [1.1, 1.2],
    [1.3, 1.4],
    [1.5, 1.6]
], dtype=np.float32)

# 1. gate_proj = input @ w3
gate_proj = np.dot(input_tensor, w3)
print(f"gate_proj: {gate_proj}")

# 2. activated_gate = SiLU(gate_proj)
activated_gate = silu(gate_proj)
print(f"activated_gate: {activated_gate}")

# 3. hidden_proj = input @ w1
hidden_proj = np.dot(input_tensor, w1)
print(f"hidden_proj: {hidden_proj}")

# 4. intermediate = hidden_proj * activated_gate (element-wise)
intermediate = hidden_proj * activated_gate
print(f"intermediate: {intermediate}")

# 5. output = intermediate @ w2
output = np.dot(intermediate, w2)
print(f"output: {output}")