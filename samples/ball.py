import numpy as np
import pyqtgraph.opengl as gl


class Ball():
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center

    def sample(self, num):
        p = []

        for i in range(num):
            z = 2 * self.radius * np.random.random_sample() - self.radius
            theta = 2 * np.pi * np.random.random_sample()
            a = np.sqrt(self.radius ** 2 - z ** 2)
            x = np.cos(theta) * a
            y = np.sin(theta) * a

            p.append(np.array([x, y, z]))

        return p

    def plotSurface(self):
        md = gl.MeshData.sphere(rows=10, cols=20, radius=self.radius)
        m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=(0.1, 0.1, 0.1, 0.2), shader='balloon', glOptions='opaque')

        m1.translate(self.center[0], self.center[1], self.center[2])

        return m1










