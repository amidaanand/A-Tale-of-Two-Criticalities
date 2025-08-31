import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

# Load MNIST data
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
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(input_dim,),
                                    activation='relu', 
                                    kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED)))
    for _ in range(num_hidden_layers - 1):
        model.add(tf.keras.layers.Dense(hidden_dim, activation='relu',
                                        kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED)))
    model.add(tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED)))
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

# Train with regime
def train_with_mode(mode, noise_std=0.05, decay_start=30, decay_rate=0.95):
    model = build_model()
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        print(f"[{mode.upper()}] Epoch {epoch+1}/{epochs}")
        
        history = model.fit(x_train, y_train,
                            validation_data=(x_test, y_test),
                            epochs=1,
                            batch_size=batch_size,
                            verbose=1)

        train_loss.append(history.history['loss'][0])
        val_loss.append(history.history['val_loss'][0])

        # Add noise or forgetting after training
        if mode == "forgetting" and epoch >= decay_start:
            for w in model.trainable_weights:
                w.assign(w * decay_rate)

        if mode == "noise":
            for w in model.trainable_weights:
                noise = tf.random.normal(w.shape, stddev=noise_std)
                w.assign_add(noise)

        if mode == "noise 30" and epoch >= decay_start:
            for w in model.trainable_weights:
                noise = tf.random.normal(w.shape, stddev=noise_std)
                w.assign_add(noise)

    return train_loss, val_loss

# Run all three regimes
modes = ["baseline", "noise", "forgetting", "noise 30"]
loss_results = {}

for mode in modes:
    train_loss, val_loss = train_with_mode(mode)
    loss_results[mode] = (train_loss, val_loss)

# Plotting all loss curves
plt.figure(figsize=(12, 6))
colors = {'baseline': 'royalblue', 'noise': 'darkorange', 'forgetting': 'forestgreen', 'noise 30': 'salmon'}

for mode in modes:
    train_loss, val_loss = loss_results[mode]
    plt.plot(train_loss, linestyle='--', color=colors[mode], label=f"{mode.capitalize()} - Train Loss")
    plt.plot(val_loss, linestyle='-', color=colors[mode], label=f"{mode.capitalize()} - Val Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves Across Baseline, Noise, Noise after 30, and Forgetting Regimes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
