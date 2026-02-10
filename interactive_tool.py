import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata, RegularGridInterpolator

# -----------------------------
# Load data from simulations
# -----------------------------
with open('summary_all.csv', 'r') as input_file:
    input_data = input_file.readlines()

combinations = np.array([
    (float(input_data[i].split(",")[1]),
     float(input_data[i].split(",")[2]) / float(input_data[i].split(",")[3]))
    for i in range(1, len(input_data))
])

segregation_strength = np.array([
    float(input_data[i].split(",")[6]) * 100.0
    for i in range(1, len(input_data))
])

# -----------------------------
# Interpolation grid
# -----------------------------
x_interp = np.linspace(np.min(combinations[:, 0]), np.max(combinations[:, 0]), 50)
y_interp = np.linspace(np.min(combinations[:6, 1]), np.max(combinations[6:, 1]), 50)
x_grid, y_grid = np.meshgrid(x_interp, y_interp)

# Linear interpolation + nearest fill (avoids NaNs outside convex hull)
z_linear = griddata(combinations, segregation_strength, (x_grid, y_grid), method="linear")
z_nearest = griddata(combinations, segregation_strength, (x_grid, y_grid), method="nearest")
z_interp = np.where(np.isnan(z_linear), z_nearest, z_linear)

# Overwrite interpolated grid with real data values
for i, (x_val, y_val) in enumerate(combinations):
    xi = np.abs(x_interp - x_val).argmin()
    yi = np.abs(y_interp - y_val).argmin()
    z_interp[yi, xi] = segregation_strength[i]

max_z = float(np.nanmax(z_interp))

# Regular grid interpolator for smooth updates at arbitrary slider positions
z_rgi = RegularGridInterpolator(
    (y_interp, x_interp),
    z_interp,
    bounds_error=False,
    fill_value=np.nan
)

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(12, 12))
fig.suptitle(
    "Predictive design of stratification in binary supraparticles",
    fontsize=18,
    weight="bold",
    y=0.98
)

gs = gridspec.GridSpec(
    5, 4, 
    height_ratios=[0.6, 0.2, 3., 0.2, 0.2],
    width_ratios=[0.2, 3., 0.8, 0.2], 
    hspace=0.15,
    wspace=0.3
)

# -----------------------------
# Main contour plot
# -----------------------------
ax_heatmap = fig.add_subplot(gs[2, 1])
heatmap = ax_heatmap.contourf(x_grid, y_grid, z_interp, levels=50, cmap="jet")
ax_heatmap.set_title("Interpolated contour plot", fontsize=15, weight="bold")
ax_heatmap.set_xlabel(r"Volume fraction of small particles $\phi_{S}$ / %", fontsize=15)
ax_heatmap.set_ylabel(r"Particle size ratio $d_{L}/d_{S}$ / -", fontsize=15)
ax_heatmap.tick_params(axis="both", labelsize=12)
ax_heatmap.set_ylim(1, 7)

# Crosshair lines that follow sliders
vline = ax_heatmap.axvline(x=float(x_interp.min()), color="red", linewidth=2, ls="--", zorder=10)
hline = ax_heatmap.axhline(y=float(y_interp.min()), color="blue", linewidth=2, ls="--", zorder=10)

# Colorbar
cbar_ax = fig.add_subplot(gs[2, 3])
cbar = plt.colorbar(heatmap, cax=cbar_ax, orientation="vertical")
cbar.set_label(r"Degree of stratification $\Delta\chi_{SE}$ / %", fontsize=15)
cbar.ax.tick_params(labelsize=12)

# -----------------------------
# Side plots
# -----------------------------
# Z vs Y for selected X (right)
ax_plot_x = fig.add_subplot(gs[2, 2])
ax_plot_x.set_xlabel(r"$\Delta\chi_{SE}$ / %", fontsize=15,)
ax_plot_x.tick_params(axis="both", labelsize=12)
ax_plot_x.set_xlim(0, max_z)
ax_plot_x.set_ylim(float(y_interp.min()), float(y_interp.max()))
line_x, = ax_plot_x.plot(z_interp[:, 0], y_interp, linewidth=2, color="red")

# Z vs X for selected Y (top)
ax_plot_y = fig.add_subplot(gs[0, 1])
ax_plot_y.set_ylabel(r"$\Delta\chi_{SE}$ / %", fontsize=15)
ax_plot_y.tick_params(axis="both", labelsize=12)
ax_plot_y.set_xlim(float(x_interp.min()), float(x_interp.max()))
ax_plot_y.set_ylim(0, max_z)
line_y, = ax_plot_y.plot(x_interp, z_interp[0, :], linewidth=2, color="blue")

# -----------------------------
# Label for side plots
# -----------------------------
# Live value label for phi_S
txt_phi = ax_plot_y.text(
    0.5, 1.02, "",
    transform=ax_plot_x.transAxes,
    ha="center", va="bottom",
    fontsize=15,
    color="red",
    clip_on=False
)

# Live value label for dL/dS
txt_ratio = ax_plot_x.text(
    1.02, 0.5, "",
    transform=ax_plot_y.transAxes,
    ha="left", va="center",
    fontsize=15,
    color="blue",
    clip_on=False
)



# -----------------------------
# Sliders
# -----------------------------
# X Slider (bottom)
ax_x = fig.add_subplot(gs[4, 1])
slider_x = Slider(
    ax=ax_x,
    label="",
    valmin=1,
    valmax=30,
    valinit=float(x_interp.min()),
    valstep=1,
    color="red"
)

# Y Slider (left, vertical)
ax_y = fig.add_subplot(gs[2, 0])
slider_y = Slider(
    ax=ax_y,
    label="",
    valmin=1,
    valmax=7,
    valinit=float(y_interp.min()),
    valstep=0.2,
    color="blue",
    orientation="vertical"
)

# -----------------------------
# Update logic (single callback for both sliders)
# -----------------------------
def update(_=None):
    current_x = float(slider_x.val)
    current_y = float(slider_y.val)

    # Update crosshair lines on contour plot
    vline.set_xdata([current_x, current_x])
    hline.set_ydata([current_y, current_y])

    # Update slice plots using regular grid interpolation
    pts_v = np.column_stack([y_interp, np.full_like(y_interp, current_x)])  # (y, x)
    z_vs_y = z_rgi(pts_v)
    line_x.set_xdata(z_vs_y)
    line_x.set_ydata(y_interp)

    pts_h = np.column_stack([np.full_like(x_interp, current_y), x_interp])  # (y, x)
    z_vs_x = z_rgi(pts_h)
    line_y.set_xdata(x_interp)
    line_y.set_ydata(z_vs_x)

    # Update description values
    txt_phi.set_text(rf"$\phi_S$ = {current_x:.0f} %")
    txt_ratio.set_text(rf"$d_L/d_S$ = {current_y:.1f}")

    fig.canvas.draw_idle()

slider_x.on_changed(update)
slider_y.on_changed(update)

# Initialize all elements to match initial slider positions
update()

plt.show()
