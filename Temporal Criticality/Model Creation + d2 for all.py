import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import Input
from keras.callbacks import Callback
from scipy.stats import zscore
from statsmodels.tsa.stattools import yule_walker
from scipy.special import comb

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
fixed_input = x_test[0:1]

# AR + d2 tools
def fit_ar_model(signal, order):
    phi, _ = yule_walker(signal, order=order, method='mle')
    return phi

def construct_fixed_point_basis(order, beta):
    m = beta // 2 - 1
    x = np.zeros((order, order - m))
    for k in range(m+1, order+1):
        for t in range(1, k+1):
            x[t-1, k - m - 1] = comb(k, t) * (-1)**(t + 1)
    return x

def compute_d_beta(phi, beta, order):
    X = construct_fixed_point_basis(order, beta)
    x0 = X[:, 0]
    X_shifted = X[:, 1:] - x0[:, None]
    model = phi.reshape(-1, 1)
    bhat = model - x0[:, None]
    Q, R = np.linalg.qr(X_shifted)
    v = np.linalg.solve(R, Q.T @ bhat)
    closest_model = x0[:, None] + X_shifted @ v
    return np.linalg.norm(model - closest_model)

def compute_d2_over_time(series, window_size=20, step_size=5, order=10):
    d2_vals, centers = [], []
    for start in range(0, len(series) - window_size + 1, step_size):
        window = series[start:start + window_size]
        if np.std(window) < 1e-8:
            continue
        z_window = zscore(window)
        phi = fit_ar_model(z_window, order)
        d2 = compute_d_beta(phi, beta=2, order=order)
        d2_vals.append(d2)
        centers.append(start + window_size // 2)
    return centers, d2_vals

# Callback for tracking population activity across modes
class PopulationLogger(Callback):
    def __init__(self, fixed_input, mode="baseline", noise_level=0.1, decay_epoch=30, decay_factor=0.95):
        self.fixed_input = fixed_input
        self.mode = mode
        self.noise_level = noise_level
        self.decay_epoch = decay_epoch
        self.decay_factor = decay_factor
        self.pop_activity = []

    def on_epoch_end(self, epoch, logs=None):
        _ = self.model(self.fixed_input, training=False)

        hidden_outputs = []
        for i in range(1, len(self.model.layers) - 1):  # exclude input and output layers
            submodel = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.layers[i].output)
            activations = submodel(self.fixed_input, training=False).numpy()
            if self.mode == "noise":
                activations += np.random.normal(0, self.noise_level, activations.shape)
            hidden_outputs.append(activations)

        full_hidden_vector = np.concatenate([act.flatten() for act in hidden_outputs])
        spike_count = np.sum(full_hidden_vector > 0)
        self.pop_activity.append(spike_count)

        if self.mode == "forgetting" and epoch >= self.decay_epoch:
            for w in self.model.trainable_weights:
                w.assign(w * self.decay_factor)

# Training function
def run_training(mode, seed=42, epochs=60, hidden_dim=512):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = Sequential()
    model.add(Input(shape=(784,)))
    for _ in range(3):
        model.add(Dense(hidden_dim, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    _ = model(fixed_input)  # Force model build

    logger = PopulationLogger(fixed_input, mode=mode)
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=32, epochs=epochs, verbose=1, callbacks=[logger])

    return logger.pop_activity

# Run across modes and seeds
modes = ["baseline", "noise", "forgetting"]
seeds = [42, 101, 202, 303, 404]
pop_series_dict = {mode: [] for mode in modes}
d2_curves_dict = {mode: [] for mode in modes}
all_centers = None

for mode in modes:
    print(f"\n🔁 Running mode: {mode}")
    for i, seed in enumerate(seeds):
        print(f"Seed {seed} ({i+1}/{len(seeds)})")
        pop_series = run_training(mode=mode, seed=seed)
        pop_series_dict[mode].append(pop_series)
        centers, d2_vals = compute_d2_over_time(pop_series)
        d2_curves_dict[mode].append(d2_vals)
        if all_centers is None:
            all_centers = centers

# Plot population activity curves
for mode in modes:
    pop_array = np.array(pop_series_dict[mode])
    pop_mean = np.mean(pop_array, axis=0)
    pop_std = np.std(pop_array, axis=0)

    plt.figure(figsize=(10, 5))
    for i, series in enumerate(pop_series_dict[mode]):
        plt.plot(range(len(series)), series, alpha=0.3)
    plt.plot(range(len(series)), pop_mean, color='black', linewidth=2, label='Mean')
    plt.xlabel("Epoch")
    plt.ylabel("Population Spike Count")
    plt.title(f"Population Activity - {mode.capitalize()}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot combined d2 curves
plt.figure(figsize=(10, 6))
for mode in modes:
    d2_array = np.array(d2_curves_dict[mode])
    d2_mean = np.mean(d2_array, axis=0)
    d2_std = np.std(d2_array, axis=0)
    plt.plot(all_centers, d2_mean, label=f"{mode.capitalize()} $d_2$")
    plt.fill_between(all_centers, d2_mean - d2_std, d2_mean + d2_std, alpha=0.3)

plt.xlabel("Epoch (center of window)")
plt.ylabel(r"$d_2$")
plt.title(r"Comparison of $d_2$ Across Baseline, Noise, and Forgetting")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

########WINDOW SIZES

window_sizes = [5, 10, 20, 30, 40]
mode = "baseline"
pop_series_list = pop_series_dict[mode]  # list of time series (one per seed)

# Step 1: Compute mean population activity over seeds
pop_array = np.array([np.asarray(s).flatten() for s in pop_series_list])
pop_mean_series = np.mean(pop_array, axis=0)

# Step 2: Compute d2 using different window sizes on this mean series
for ws in window_sizes:
    print(f"🔁 Running ws={ws}")
    centers, d2_vals = compute_d2_over_time(pop_mean_series, window_size=ws, step_size=5, order=10)
    plt.plot(centers, d2_vals, label=f'ws={ws}')

# Step 3: Plot
plt.xlabel("Epoch (center of window)")
plt.ylabel(r"$d_2$ (Distance to $\beta=2$ criticality)")
plt.title(r"$d_2$ Across Window Sizes (Mean Population Activity – Baseline)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
