import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class QuadcopterRenderer():

    def __init__(self, quadcopter):
        plt.rcParams["font.family"] = "monospace"
        plt.rcParams["font.size"] = 6
        plt.rcParams['lines.linewidth'] = 2

        self.quadcopter = quadcopter

        # Plot
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
        self.fig.set_dpi(400) # Controls output resolution

    def render(self, title=None, label=None):
        # Clear plot and reset axes properties
        self.ax.clear()
        self.ax.set(xlim3d=(-1, 1), xlabel='x')
        self.ax.set(ylim3d=(-1, 1), ylabel='y')
        self.ax.set(zlim3d=(-1, 1), zlabel='z')

        # Model shape of quadcopter with rotation
        l = self.quadcopter.l # Arm length
        s = l * np.sin(self.quadcopter.state[2])
        c = l * np.cos(self.quadcopter.state[2])
        self.model = [[0, l, 0, 0, 0, -l, 0, 0, 0],
                      [0, 0, 0, c, 0, 0, 0, -c, 0],
                      [0, 0, 0, s, 0, 0, 0, -s, 0]]

        # Translate quadcopter model with position
        self.ax.plot(np.add(self.model[0], 0),
                     np.add(self.model[1], self.quadcopter.state[0]),
                     np.add(self.model[2], self.quadcopter.state[1]))

        # Annotate figure with stationary and label text
        axto = (0.1, 0.1, 0) # ax text offset. Merely for display purposes
        plt.title(title, loc="left")
        self.ax.text(axto[0], self.quadcopter.state[0] + axto[1], self.quadcopter.state[1] + axto[2], label,
                ha="left",
                va="center",
                bbox=dict(boxstyle="square",
                          facecolor="xkcd:baby shit green",
                          alpha=0.4
                    )
            )

        # Render the figure into an in-memory PNG.
        buffer = io.BytesIO()
        self.fig.savefig(
            buffer,
            format="png",
            dpi=self.fig.dpi,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        buffer.seek(0)

        image = Image.open(buffer).convert("RGB")
        frame = np.asarray(image, dtype=np.uint8).copy()

        buffer.close()

        return frame
