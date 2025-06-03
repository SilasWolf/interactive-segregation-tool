import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from scipy.interpolate import RBFInterpolator
import mpld3
from scipy.interpolate import griddata
import os

# Load data from simulations

with open('summary_all.csv', 'r') as input_file:
    input_data = input_file.readlines()

combinations = np.array([
    (float(input_data[i].split(',')[1]),
     float(input_data[i].split(',')[2]) / float(input_data[i].split(',')[3]))
    for i in range(1, len(input_data))])

segregation_strength = np.array([float(input_data[i].split(',')[6]) for i in range(1, len(input_data))])

# Interpolation Grid
x_interp = np.linspace(np.min(combinations[:,0]), np.max(combinations[:,0]), 50)  # More refined X values
y_interp = np.linspace(np.min(combinations[:,1]), np.max(combinations[:,1]), 50)  # More refined Y values
x_grid, y_grid = np.meshgrid(x_interp, y_interp)
print(combinations)
# Interpolating the Z values
z_interp = griddata(combinations, segregation_strength, (x_grid, y_grid), method='linear')

# Extrapolate for regions outside the max values
rbf = RBFInterpolator(combinations, segregation_strength, kernel='linear')
z_rbf = rbf(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

# Replace only NaNs from original linear interpolation outside of the max values
z_interp[np.isnan(z_interp)] = z_rbf[np.isnan(z_interp)]

# Create figure and GridSpec layout

fig = plt.figure(figsize=(12, 12))  # Keep figure size large for better spacing
gs = gridspec.GridSpec(
    5, 4, 
    height_ratios=[0.6, 0.2, 3., 0.2, 0.2],  # Ensure sliders are of equal thickness (0.2 each)
    width_ratios=[0.2, 3., 0.8, 0.2],  # Keep wider 2D plot
    hspace=0.15,  # Reduce vertical spacing
    wspace=0.3
)

# Heatmap with Interpolated Data
ax_heatmap = fig.add_subplot(gs[2, 1])
heatmap = ax_heatmap.contourf(x_grid, y_grid, z_interp, levels=50, cmap='jet')
ax_heatmap.set_title('Interpolated contour plot', fontsize=10)
ax_heatmap.set_xlabel('Volume ratio / -', fontsize=10)
ax_heatmap.set_ylabel('Size ratio / -', fontsize=10)
ax_heatmap.tick_params(axis='both', labelsize=8)

# Colorbar
cbar_ax = fig.add_subplot(gs[2, 3])
cbar = plt.colorbar(heatmap, cax=cbar_ax, orientation='vertical')
cbar.set_label('Enrichment difference / -', fontsize=10)

max_z = np.nanmax(z_interp)

# Z vs Y for selected X
ax_plot_x = fig.add_subplot(gs[2, 2])
#ax_plot_x.set_title( fontsize=10)
ax_plot_x.set_xlabel('Enrichment difference / -', fontsize=10)
#ax_plot_x.set_ylabel('Size ratio / -', fontsize=10)
ax_plot_x.tick_params(axis='both', labelsize=8)
ax_plot_x.set_xlim(0, max_z)
ax_plot_x.set_ylim(min(y_interp), max(y_interp))
line_x, = ax_plot_x.plot(z_interp[:, 0], y_interp, linewidth=2, color='red')

# Z vs X for selected Y
ax_plot_y = fig.add_subplot(gs[0, 1])
#ax_plot_y.set_title('Relative segregation over volume ratio\nfor selected size ratio', fontsize=10)
#ax_plot_y.set_xlabel('Volume ratio / -', fontsize=10)
ax_plot_y.set_ylabel('Enrichment difference / -', fontsize=10)
ax_plot_y.tick_params(axis='both', labelsize=8)
ax_plot_y.set_xlim(min(x_interp), max(x_interp))
ax_plot_y.set_ylim(0, max_z)
line_y, = ax_plot_y.plot(x_interp, z_interp[0, :], linewidth=2,color='blue')

# X Slider
ax_x = fig.add_subplot(gs[4, 1])
slider_x = Slider(ax=ax_x, valmin=x_interp.min(), label='',valmax=x_interp.max(), valinit=x_interp.min(), valstep=x_interp[1] - x_interp[0], color='red')

# Y Slider (Vertical)
ax_y = fig.add_subplot(gs[2, 0])
slider_y = Slider(ax=ax_y,  valmin=y_interp.min(), label='',valmax=y_interp.max(), valinit=y_interp.min(), valstep=y_interp[1] - y_interp[0], color='blue', orientation='vertical')

# Update function for X
def update_x(val):
    current_x = slider_x.val
    idx = np.searchsorted(x_interp, current_x)
    if idx >= len(x_interp):
        idx = len(x_interp) - 1
    current_z = z_interp[:, idx]  # Interpolated Z values for selected X
    line_x.set_xdata(current_z)
    line_x.set_ydata(y_interp)
    fig.canvas.draw_idle()

# Update function for Y
def update_y(val):
    current_y = slider_y.val
    idy = np.searchsorted(y_interp, current_y)
    if idy >= len(y_interp):
        idy = len(y_interp) - 1
    current_z = z_interp[idy, :]  # Interpolated Z values for selected Y
    line_y.set_xdata(x_interp)
    line_y.set_ydata(current_z)
    fig.canvas.draw_idle()

slider_x.on_changed(update_x)
slider_y.on_changed(update_y)

plt.show()
