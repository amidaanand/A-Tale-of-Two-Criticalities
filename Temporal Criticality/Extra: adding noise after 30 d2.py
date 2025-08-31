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

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
fixed_input = x_test[0:1]

# AR tools
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

# Logger
class PopulationLogger(Callback):
    def __init__(self, fixed_input, mode="baseline", noise_epoch=30, noise_level=0.1):
        self.fixed_input = fixed_input
        self.mode = mode
        self.noise_epoch = noise_epoch
        self.noise_level = noise_level
        self.pop_activity = []

    def on_epoch_end(self, epoch, logs=None):
        _ = self.model(self.fixed_input, training=False)
        hidden_outputs = []
        for i in range(1, len(self.model.layers) - 1):
            submodel = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.layers[i].output)
            activations = submodel(self.fixed_input, training=False).numpy()
            if self.mode == "noise" or (self.mode == "noise_late" and epoch >= self.noise_epoch):
                activations += np.random.normal(0, self.noise_level, activations.shape)
            hidden_outputs.append(activations)
        full_vector = np.concatenate([act.flatten() for act in hidden_outputs])
        spike_count = np.sum(full_vector > 0)
        self.pop_activity.append(spike_count)

# Training function
def run_training(mode, seed=42, epochs=60):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = Sequential()
    model.add(Input(shape=(784,)))
    for _ in range(3):
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    _ = model(fixed_input)
    logger = PopulationLogger(fixed_input, mode=mode)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
              batch_size=32, verbose=1, callbacks=[logger])
    return logger.pop_activity

# Run
modes = ["baseline", "noise", "noise_late"]
seeds = [42, 101, 202]
pop_data = {mode: [] for mode in modes}
d2_data = {mode: [] for mode in modes}
all_centers = None

for mode in modes:
    for seed in seeds:
        print(f"Running {mode} with seed {seed}")
        spikes = run_training(mode, seed)
        pop_data[mode].append(spikes)
        centers, d2_vals = compute_d2_over_time(spikes)
        d2_data[mode].append(d2_vals)
        if all_centers is None:
            all_centers = centers

# Plot population activity
for mode in modes:
    spikes = np.array(pop_data[mode])
    mean = np.mean(spikes, axis=0)
    std = np.std(spikes, axis=0)
    plt.figure(figsize=(8, 5))
    for run in spikes:
        plt.plot(run, alpha=0.3)
    plt.plot(mean, label=f"{mode} mean", color='black')
    plt.xlabel("Epoch")
    plt.ylabel("Population Activity")
    plt.title(f"{mode.capitalize()} - Hidden Neuron Activations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot d2 curves
plt.figure(figsize=(10, 6))
for mode in modes:
    d2_array = np.array(d2_data[mode])
    mean = np.mean(d2_array, axis=0)
    std = np.std(d2_array, axis=0)
    plt.plot(all_centers, mean, label=f"{mode} $d_2$")
    plt.fill_between(all_centers, mean - std, mean + std, alpha=0.3)
plt.xlabel("Epoch (center of window)")
plt.ylabel(r"$d_2$")
plt.title("Comparison of $d_2$ Across Baseline, Noise, and Noise-After-30")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
