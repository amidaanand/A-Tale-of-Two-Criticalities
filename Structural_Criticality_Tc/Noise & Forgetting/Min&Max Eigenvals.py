modes_to_compare = ['baseline', 'noise', 'forgetting']

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

plt.figure(figsize=(10, 6))

for mode, color in zip(modes_to_compare, ['purple', 'salmon', 'green']):
    epochs_tracked = []
    eig_min = []
    eig_max = []

    for epoch, eigvals_sorted in spectra_results[mode]:
        epochs_tracked.append(epoch)
        eig_min.append(np.min(eigvals_sorted))
        eig_max.append(np.max(eigvals_sorted))

    plt.plot(epochs_tracked, eig_max, 'o-', label=f'{mode.capitalize()} Max Eigenvalue', color=color)
    plt.plot(epochs_tracked, eig_min, 's--', label=f'{mode.capitalize()} Min Eigenvalue', color=color, alpha=0.7)

epochs_tracked_30 = []
eig_min_30 = []
eig_max_30 = []

for epoch, eigvals_sorted in spectra:
    epochs_tracked_30.append(epoch)
    eig_min_30.append(np.min(eigvals_sorted))
    eig_max_30.append(np.max(eigvals_sorted))

plt.plot(epochs_tracked, eig_max, 'o-', label="Noise 30 Max Eigenvalue", color='gold')
plt.plot(epochs_tracked, eig_min, 's-', label="Noise 30 Min Eigenvalue", color='gold')

plt.title("Comparison of Max & Min Eigenvalues")
plt.xlabel("Epoch")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

######COMPARISON BETWEEN NOISE AND FORGETTING

modes_to_compare = ['forgetting']

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

plt.figure(figsize=(10, 6))

for mode, color in zip(modes_to_compare, ['green']):
    epochs_tracked = []
    eig_min = []
    eig_max = []

    for epoch, eigvals_sorted in spectra_results[mode]:
        epochs_tracked.append(epoch)
        eig_min.append(np.min(eigvals_sorted))
        eig_max.append(np.max(eigvals_sorted))

    plt.plot(epochs_tracked, eig_max, 'o-', label=f'{mode.capitalize()} Max Eigenvalue', color=color)
    plt.plot(epochs_tracked, eig_min, 's--', label=f'{mode.capitalize()} Min Eigenvalue', color=color, alpha=0.7)

epochs_tracked_30 = []
eig_min_30 = []
eig_max_30 = []

for epoch, eigvals_sorted in spectra:
    epochs_tracked_30.append(epoch)
    eig_min_30.append(np.min(eigvals_sorted))
    eig_max_30.append(np.max(eigvals_sorted))

plt.plot(epochs_tracked, eig_max, 'o-', label="Noise 30 Max Eigenvalue", color='gold')
plt.plot(epochs_tracked, eig_min, 's-', label="Noise 30 Min Eigenvalue", color='gold')

plt.title("Comparison of Max & Min Eigenvalues for Forgetting and Noise (after 30)")
plt.xlabel("Epoch")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
