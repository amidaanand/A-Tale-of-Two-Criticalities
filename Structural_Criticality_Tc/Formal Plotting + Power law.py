import matplotlib.pyplot as plt

# Professional style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.size": 12,
    "text.usetex": False,        # don't call external LaTeX
    "font.family": "serif",      
    "mathtext.fontset": "cm",    # use mathtext's Computer Modern
})



main_color = "indigo"   
grid_color = "#E0E0E0"   # subtle grey

fig, ax = plt.subplots(figsize=(8,5))
    

plt.plot(epoch_checkpoints, tc_mean, 'o-', label='Mean $T_c$', color=main_color)
plt.fill_between(epoch_checkpoints, tc_mean - tc_std, tc_mean + tc_std, color='purple', alpha=0.3)

# Main line
'''ax.plot(epoch_checkpoints, T_c_evolution,
        'o-', color=main_color, linewidth=2, markersize=5,
        label=r"$T_c$ over training")'''

# Axis labels & title
ax.set_xlabel(r"$\mathrm{Epoch}$")     # 'Epoch' looks LaTeX-y now
ax.set_ylabel(r"$T_c$")
ax.set_title(r"$\mathrm{Evolution\ of\ Critical\ Temperature}$")

# Grid
ax.grid(color=grid_color, linestyle='-', linewidth=0.5)

# Legend
ax.legend()

# Thinner, dark grey spines
for spine in ax.spines.values():
    spine.set_color("black")

plt.tight_layout()
plt.show()

#####POWER LAW FITTING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# --- Styling (same as first plot) ---
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.size": 12,
    "text.usetex": False,        # using mathtext, not external LaTeX
    "font.family": "serif",      
    "mathtext.fontset": "cm",    # Computer Modern
})

main_color = "indigo"    # keep consistent with first plot
fit_color1 = "#F418D3"   # muted red for early fit
fit_color2 = "#EBDC5B"   # muted blue for late fit
grid_color = "#E0E0E0"   # subtle grey grid

# --- Power-law model ---
def power_law(x, A, alpha):
    return A * (x ** alpha)

# Convert to arrays
x = np.array(epoch_checkpoints)
y = np.array(tc_mean)

# Define early and late regions
early_mask = (x >= 2) & (x <= 30)
late_mask = x >= 25

x_early, y_early = x[early_mask], y[early_mask]
x_late, y_late = x[late_mask], y[late_mask]

# Fit both regions
popt_early, _ = curve_fit(power_law, x_early, y_early)
popt_late, _ = curve_fit(power_law, x_late, y_late)

A_early, alpha_early = popt_early
A_late, alpha_late = popt_late

# --- Plot ---
fig, ax = plt.subplots(figsize=(8,5))

# Data points
ax.plot(x, y, 'o-', color=main_color, linewidth=2, markersize=5, label=r"$T_c$ data")

# Fitted power-law (early region)
ax.plot(x_early, power_law(x_early, A_early, alpha_early), 
        linestyle="--", color=fit_color1, linewidth=2,
        label=fr"${A_early:.2f} \, t^{{{alpha_early:.3f}}}$")

# Fitted power-law (late region)
ax.plot(x_late, power_law(x_late, A_late, alpha_late), 
        linestyle="--", color=fit_color2, linewidth=2,
        label=fr"${A_late:.2f} \, t^{{{alpha_late:.4f}}}$")

# Log scales
ax.set_xscale("log")
ax.set_yscale("log")


# Axis labels & title
ax.set_xlabel(r"$\mathrm{Epoch}$")     # 'Epoch' looks LaTeX-y now
ax.set_ylabel(r"$T_c$")
ax.set_title(r"$\mathrm{Log-log\ Plot\ of\ Critical\ Temperature\ with\ Power-Law\ Fits}$")

# Grid and legend
ax.grid(color=grid_color, linestyle='-', linewidth=0.5, which="both")
ax.legend(frameon=False)

# Clean spines
for spine in ax.spines.values():
    spine.set_color("black")

plt.tight_layout()
plt.show()
