import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Load the cube
cube = np.load("/Users/giannigagliardi/Documents/Git/RadioTherapy/traindata/11_5/outputcube/235101017859661465075472232303048949736_10.npy")
# If the loaded cube has an extra energy dimension, pick the first energy
if cube.ndim == 4:
    cube = cube[0]
elif cube.ndim != 3:
    raise ValueError(f"Unexpected cube shape {cube.shape}; expected 3D or 4D (energy × D × H × W).")
# ---------- Interactive slice viewer ----------
import matplotlib.widgets as widgets

fig, ax = plt.subplots()
slice_idx = cube.shape[2] // 2
im = ax.imshow(cube[:, :, slice_idx], cmap="gray")
ax.set_title(f"Slice {slice_idx}")
ax.axis("off")

slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = widgets.Slider(
    slider_ax, "z‑Slice", 0, cube.shape[2] - 1,
    valinit=slice_idx, valfmt="%0.0f"
)

def update(val):
    z = int(slider.val)
    im.set_data(cube[:, :, z])
    ax.set_title(f"Slice {z}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
# ---------- end slice viewer ----------

cube_norm = (cube - cube.min()) / (cube.max() - cube.min())
# --- choose sensible isosurface thresholds ---------------------------------
# This prints the 1st and 99th percentiles so we know where most intensities lie.
p_low, p_high = np.percentile(cube_norm, [1, 99])
print(f"Suggested thresholds  p_low={p_low:.4f}  p_high={p_high:.4f}")

# create 3‑D coordinate grid that matches the cube dimensions
x, y, z = np.mgrid[0:cube.shape[0], 0:cube.shape[1], 0:cube.shape[2]]

# ----------- 3‑D volume rendering -------------
# Apply smoothing to reduce artifacts
from scipy import ndimage
cube_smoothed = ndimage.gaussian_filter(cube_norm, sigma=0.5)

# If the printed p_low/p_high look too small or too large,
# feel free to tweak them below.
fig_vol = go.Figure(
    data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=cube_smoothed.flatten(),
        opacity=0.1,               # mehr transparent für bessere Sicht
        surface_count=15,          # weniger Isoflächen für glatteres Aussehen
        isomin=p_low * 1.2,        # etwas höhere Schwelle
        isomax=p_high * 0.8,       # etwas niedrigere obere Schwelle
        colorscale="Hot",          # bessere Farbskala für Dosisverteilung
        caps=dict(x_show=False, y_show=False, z_show=False)  # keine Kappen
    )
)
# realistische Seitenverhältnisse beibehalten
fig_vol.update_layout(scene=dict(aspectmode="data"))
# enforce axis ranges to match cube shape
fig_vol.update_layout(
    scene=dict(
        xaxis=dict(range=[0, cube.shape[0]]),
        yaxis=dict(range=[0, cube.shape[1]]),
        zaxis=dict(range=[0, cube.shape[2]])
    )
)
fig_vol.show()
# ----------- end volume rendering -------------