import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class QuadcopterRenderer():

    def __init__(self, quadcopter):
        # Model shape of quadcopter
        l = quadcopter.l # Arm length
        self.model = [[0, l, 0, 0, 0, -l, 0, 0, 0],
                      [0, 0, 0, l, 0, 0, 0, -l, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        plt.style.use('_mpl-gallery')

        # Plot
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
        self.fig.set_dpi(400) # Controls output resolution

    def render(self, position, rotation):
        # Clear plot and reset axes properties
        self.ax.clear()
        self.ax.set(xlim3d=(-1, 1), xlabel='x')
        self.ax.set(ylim3d=(-1, 1), ylabel='y')
        self.ax.set(zlim3d=(-1, 1), zlabel='z')

        # Translate quadcopter model with position
        self.ax.plot(np.add(self.model[0], position[0]),
                     np.add(self.model[1], position[1]),
                     np.add(self.model[2], position[2]))

        # Render the figure into an in-memory PNG.
        buffer = io.BytesIO()
        self.fig.savefig(
            buffer,
            format="png",
            dpi=self.fig.dpi,
            bbox_inches=None,
            pad_inches=0,
        )

        buffer.seek(0)

        image = Image.open(buffer).convert("RGB")
        frame = np.asarray(image, dtype=np.uint8).copy()

        buffer.close()

        return frame
