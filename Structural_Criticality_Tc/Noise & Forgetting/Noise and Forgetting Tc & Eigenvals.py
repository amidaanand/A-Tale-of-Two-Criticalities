######THIS IS FOR NOISE ADDED FROM THE BEGINNING, WHERE TC WAS UNABLE TO BE COMPUTED 
'''import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"'''

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
SEED = 42

# Build model

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(input_dim,),
                                    activation='relu', 
                                    kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED)))
    for _ in range(num_hidden_layers - 1):  # total 3 layers
        model.add(tf.keras.layers.Dense(hidden_dim, activation='relu',
                                        kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED)))
    model.add(tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED)))
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

# Plotting helper
def plot_eigenvalue_spectrum(J, epoch, all_spectra):
    eigvals_sorted = np.sort(np.linalg.eigvalsh(J))[::-1]
    all_spectra.append((epoch, eigvals_sorted))

# Training with different regimes
def train_with_mode(mode, noise_std=0.05, decay_start=30, decay_rate=0.95):
    model = build_model()
    beta_c_evolution, T_c_evolution, checkpoints, spectra = [], [], [], []

    for epoch in range(epochs):
        print(f"[{mode.upper()}] Epoch {epoch+1}/{epochs}")
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size,
                  validation_data=(x_test, y_test), verbose=1)

        if mode == "forgetting" and epoch >= decay_start:
            for w in model.trainable_weights:
                w.assign(w * decay_rate)

        if mode == "noise":
            for w in model.trainable_weights:
                noise = tf.random.normal(w.shape, stddev=noise_std)
                w.assign_add(noise)

        weights = [layer.get_weights()[0] for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        beta_c, T_c = compute_beta_c(weights)
        
        if beta_c:
            beta_c_evolution.append(beta_c)
            T_c_evolution.append(T_c)
            checkpoints.append(epoch + 1)

        if epoch in [0, 10, 20, 30, 40, 59]:
            J = construct_J(weights)
            plot_eigenvalue_spectrum(J, epoch + 1, spectra)

    return checkpoints, T_c_evolution, spectra

# Run all modes
modes = ["baseline", "noise", "forgetting"]
tc_results = {}
spectra_results = {}

for mode in modes:
    checkpoints, T_c, spectra = train_with_mode(mode)
    tc_results[mode] = (checkpoints, T_c)
    spectra_results[mode] = spectra

# Plot all Tc curves together
plt.figure(figsize=(10, 6))
colors = {'baseline': 'black', 'noise': 'teal', 'forgetting': 'maroon'}

for mode in modes:
    x, y = tc_results[mode]
    plt.plot(x, y, 'o-', label=f"{mode.capitalize()}", color=colors[mode])

plt.xlabel("Epoch")
plt.ylabel(r"$T_c$")
plt.title("Comparison of $T_c$ Evolution Across Conditions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot eigenvalue spectra per mode
for mode in modes:
    plt.figure(figsize=(10, 5))
    for epoch, eigvals_sorted in spectra_results[mode]:
        plt.plot(eigvals_sorted, label=f"Epoch {epoch}", alpha=0.6)
    plt.title(f"Eigenvalue Spectrum: {mode.capitalize()}")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#######ADDING NOISE AFTER 30 EPOCHS

import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf

# --- Data ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# --- Parameters ---
input_dim = 784
output_dim = 10
hidden_dim = 512
epochs = 60
batch_size = 32
learning_rate = 1e-4
SEED = 42
noise_epoch_threshold = 30
noise_std = 0.02

# --- Model Setup ---
def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for _ in range(3):  # match d2 architecture
        model.add(tf.keras.layers.Dense(hidden_dim, activation='relu', 
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED)))
    model.add(tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED)))
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

# --- Construct J ---
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

# --- Compute beta_c and Tc ---
def compute_beta_c(weights):
    J = construct_J(weights)
    beta_vals = np.linspace(0.1, 1.5, 20)
    lambda_mins = []
    for beta in beta_vals:
        diag_corr = np.sum(J**2, axis=1) * np.eye(J.shape[0])
        M = beta * J - beta**2 * diag_corr
        M_prime = np.eye(J.shape[0]) - M
        eigenvals_real = eigvals(M_prime).real
        lambda_mins.append(np.min(eigenvals_real))

    for i in range(len(beta_vals) - 1):
        if lambda_mins[i] > 0 and lambda_mins[i+1] < 0:
            # Interpolate to find β_c
            beta_c = beta_vals[i] - lambda_mins[i] * (beta_vals[i+1] - beta_vals[i]) / (lambda_mins[i+1] - lambda_mins[i])
            return beta_c, 1 / beta_c
    return None, None  # No valid Tc

# --- Plot helper ---
def plot_eigen_spectrum(J, epoch, spectra_list):
    eig_sorted = np.sort(np.linalg.eigvalsh(J))[::-1]
    spectra_list.append((epoch, eig_sorted))

# --- Training ---
model = build_model()
Tc_list = []
epoch_list = []
spectra = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test), verbose=1)

    weights = [layer.get_weights()[0] for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    # Inject noise after threshold
    if epoch + 1 > noise_epoch_threshold:
        for i in range(len(weights)):
            noise = np.random.normal(0, noise_std, size=weights[i].shape)
            weights[i] += noise
            model.layers[i].set_weights([weights[i], model.layers[i].get_weights()[1]])

    # Compute Tc
    beta_c, Tc = compute_beta_c(weights)
    if Tc:
        Tc_list.append(Tc)
        epoch_list.append(epoch + 1)

    # Save spectra
    if epoch in [0, 10, 20, 30, 40, 59]:
        J = construct_J(weights)
        plot_eigen_spectrum(J, epoch + 1, spectra)

# --- Styling (same as first plot) ---
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.size": 10,
    "text.usetex": False,        # using mathtext, not external LaTeX
    "font.family": "serif",      
    "mathtext.fontset": "cm",    # Computer Modern
})

# --- Plot Tc Evolution ---
plt.figure(figsize=(8, 5))
plt.plot(epoch_list, Tc_list, 'o-', color='purple', label=r"$T_c$")
plt.axvline(noise_epoch_threshold, color='gray', linestyle='--', label="Noise Start")
plt.xlabel("Epoch")
plt.ylabel(r"$T_c$")
plt.title(r"Transition Temperature $T_c$ Evolution with Noise (after Epoch 30)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot Eigen Spectra ---
plt.figure(figsize=(10, 5))
for epoch, eigvals_sorted in spectra:
    plt.plot(eigvals_sorted, label=f"Epoch {epoch}")
plt.title("Eigenvalue Spectrum: Noise After Epoch 30")
plt.xlabel("Eigenvalue Index")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Max & Min Eigenvalues Over Epochs ---
epochs_tracked = []
eig_min = []
eig_max = []

for epoch, eigvals_sorted in spectra:
    epochs_tracked.append(epoch)
    eig_min.append(np.min(eigvals_sorted))
    eig_max.append(np.max(eigvals_sorted))

plt.figure(figsize=(8, 5))
plt.plot(epochs_tracked, eig_max, 'o-', label="Max Eigenvalue", color='navy')
plt.plot(epochs_tracked, eig_min, 'o-', label="Min Eigenvalue", color='crimson')
plt.xlabel("Epoch")
plt.ylabel("Eigenvalue")
plt.title("Evolution of Max and Min Eigenvalues Over Epochs when Noise is Added after epoch 30")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
