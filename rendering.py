import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def render(): # TODO: render quadcopter motion
    plt.style.use('_mpl-gallery')

    # Make data
    n = 100
    xs = np.linspace(0, 1, n)
    ys = np.sin(xs * 6 * np.pi)
    zs = np.cos(xs * 6 * np.pi)

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(xs, ys, zs)

    ax.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])

    # Render the figure into an in-memory PNG.
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=fig.dpi,
        bbox_inches=None,
        pad_inches=0,
    )

    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")
    frame = np.asarray(image, dtype=np.uint8).copy()

    buffer.close()

    return frame
