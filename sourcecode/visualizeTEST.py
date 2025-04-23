import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

cube = np.load("mein_cube4.npy")
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
# If the printed p_low/p_high look too small or too large,
# feel free to tweak them below.
fig_vol = go.Figure(
    data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=cube_norm.flatten(),
        opacity=0.2,               # etwas weniger durchsichtig
        surface_count=20,          # mehr Isoflächen
        isomin=p_low,
        isomax=p_high,
        colorscale="Viridis"
    )
)
# realistische Seitenverhältnisse beibehalten
fig_vol.update_layout(scene=dict(aspectmode="data"))
fig_vol.show()
# ----------- end volume rendering -------------
