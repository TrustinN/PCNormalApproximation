import numpy as np
import pyqtgraph.opengl as gl


class Plane():
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center
        self.area = np.pi * self.radius ** 2
        self.noiseFactor = self.area / 500

    def sample(self, num):
        p = []

        for i in range(num):
            theta = 2 * np.pi * np.random.random_sample()
            x = np.cos(theta) * self.radius * np.sqrt(np.random.random_sample())
            y = np.sin(theta) * self.radius * np.sqrt(np.random.random_sample())

            p.append(np.array([x, y, 0]))

        return p

    def sampleNoise(self, num):
        p = []

        for i in range(num):
            theta = 2 * np.pi * np.random.random_sample()
            x = np.cos(theta) * self.radius * np.sqrt(np.random.random_sample())
            y = np.sin(theta) * self.radius * np.sqrt(np.random.random_sample())

            p.append(np.array([x + self.noiseFactor * np.random.random_sample() - self.noiseFactor / 2,
                               y + self.noiseFactor * np.random.random_sample() - self.noiseFactor / 2,
                               self.noiseFactor * np.random.random_sample() - self.noiseFactor / 2]))

        return p

    def plotSurface(self):
        md = gl.MeshData.sphere(rows=10, cols=20, radius=self.radius)
        m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=(0.1, 0.1, 0.1, 0.2), shader='balloon', glOptions='opaque')

        m1.translate(self.center[0], self.center[1], self.center[2])

        return m1
















