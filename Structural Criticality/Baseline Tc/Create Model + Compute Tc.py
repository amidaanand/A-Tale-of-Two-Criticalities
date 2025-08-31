import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from keras.datasets import mnist
from keras.utils import to_categorical

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model parameters
input_dim = 784
output_dim = 10
hidden_dim = 512
num_hidden_layers = 3
learning_rate = 1e-4
batch_size = 32
epochs = 60

# Build model
def build_model(seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(input_dim,),
                                    activation='relu', 
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
    for _ in range(num_hidden_layers - 1):
        model.add(tf.keras.layers.Dense(hidden_dim, activation='relu',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
    model.add(tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

# Construct full J matrix
def construct_J(weights):
    N = sum(w.shape[0] for w in weights) + weights[-1].shape[1]
    J = np.zeros((N, N))
    start = 0
    for w in weights:
        rows, cols = w.shape
        J[start:start+rows, start+rows:start+rows+cols] = w
        J[start+rows:start+rows+cols, start:start+rows] = w.T
        start += rows
    return J

# Compute beta_c and T_c
def compute_beta_c(weights):
    J = construct_J(weights)
    beta_test_values = np.linspace(0.1, 1.5, 20)
    lambda_min_values = []

    for beta in beta_test_values:
        diag_corr = np.sum(J**2, axis=1) * np.eye(J.shape[0])
        M = beta * J - beta**2 * diag_corr
        M_prime = np.eye(J.shape[0]) - M
        eigenvalues = eigvals(M_prime)
        lambda_min = np.min(eigenvalues.real)
        lambda_min_values.append(lambda_min)

    beta_c, T_c = None, None
    for i in range(len(beta_test_values) - 1):
        if lambda_min_values[i] > 0 and lambda_min_values[i + 1] < 0:
            beta_c = beta_test_values[i] - lambda_min_values[i] * (
                (beta_test_values[i + 1] - beta_test_values[i]) /
                (lambda_min_values[i + 1] - lambda_min_values[i])
            )
            break

    if beta_c:
        T_c = 1 / beta_c
    return beta_c, T_c

# Run across multiple seeds
seeds = [42, 101, 202, 303, 404]
tc_curves = []
epoch_checkpoints = None

for seed in seeds:
    model = build_model(seed)
    T_c_evolution = []
    epoch_ticks = []

    for epoch in range(epochs):
        print(f"Seed {seed} - Epoch {epoch+1}/{epochs}")
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size,
                  validation_data=(x_test, y_test), verbose=1)

        weights = [layer.get_weights()[0] for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        _, T_c = compute_beta_c(weights)

        if T_c:
            T_c_evolution.append(T_c)
            epoch_ticks.append(epoch + 1)

    tc_curves.append(T_c_evolution)
    if epoch_checkpoints is None or len(epoch_ticks) < len(epoch_checkpoints):
        epoch_checkpoints = epoch_ticks

# Convert to array and compute mean/std
tc_array = np.array([curve[:len(epoch_checkpoints)] for curve in tc_curves])
tc_mean = np.mean(tc_array, axis=0)
tc_std = np.std(tc_array, axis=0)

# Plot Tc evolution
plt.figure(figsize=(8, 5))
plt.plot(epoch_checkpoints, tc_mean, 'o-', label='Mean $T_c$', color='darkred')
plt.fill_between(epoch_checkpoints, tc_mean - tc_std, tc_mean + tc_std, color='salmon', alpha=0.3)
plt.xlabel("Epoch")
plt.ylabel(r"$T_c$")
plt.title("Average Evolution of $T_c$ Across Seeds")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
