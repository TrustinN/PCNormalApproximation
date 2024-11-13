import numpy as np
from rtrees.plot import plot_mesh
import pyqtgraph.opengl as gl


class Box():
    def __init__(self, center, x, y, z):
        self.min_x = center[0] - x
        self.max_x = center[0] + x
        self.min_y = center[1] - y
        self.max_y = center[1] + y
        self.min_z = center[2] - z
        self.max_z = center[2] + z
        self.lengths = [self.max_x - self.min_x,
                        self.max_y - self.min_y,
                        self.max_z - self.min_z]
        self.margin = sum(self.lengths)
        self.ranges = np.array([[self.min_x, self.max_x],
                                [self.min_y, self.max_y],
                                [self.min_z, self.max_z]])

    def sample(self, num):
        samples = []

        for i in range(num):
            c = np.random.choice([0, 1, 2], size=2, replace=False, p=[ln / self.margin for ln in self.lengths])
            c_comp = 3 - (c[0] + c[1])

            s = self.ranges[c]
            s_comp = self.ranges[c_comp]

            p_rand = np.array([0., 0., 0.])
            p_rand[c] = np.array([(r[1] - r[0]) * np.random.random_sample() + r[0] for r in s])
            p_rand[c_comp] = s_comp[np.random.randint(2, size=1)]

            samples.append(p_rand)

        return samples

    def get_facets(self):
        vertices = np.array([
                np.array([self.min_x + 0.01, self.min_y + 0.01, self.min_z + 0.01]),  # 0
                np.array([self.max_x - 0.01, self.min_y + 0.01, self.min_z + 0.01]),  # 1
                np.array([self.min_x + 0.01, self.max_y - 0.01, self.min_z + 0.01]),  # 2
                np.array([self.min_x + 0.01, self.min_y + 0.01, self.max_z - 0.01]),  # 3
                np.array([self.max_x - 0.01, self.max_y - 0.01, self.min_z + 0.01]),  # 4
                np.array([self.min_x + 0.01, self.max_y - 0.01, self.max_z - 0.01]),  # 5
                np.array([self.max_x - 0.01, self.min_y + 0.01, self.max_z - 0.01]),  # 6
                np.array([self.max_x - 0.01, self.max_y - 0.01, self.max_z - 0.01]),  # 7
                ])
        return [
                # bottom plane
                vertices[np.array([0, 1, 4])],
                vertices[np.array([0, 4, 2])],

                # left plane
                vertices[np.array([0, 1, 6])],
                vertices[np.array([0, 6, 3])],

                # back plane
                vertices[np.array([0, 2, 5])],
                vertices[np.array([0, 5, 3])],

                # right plane
                vertices[np.array([2, 7, 5])],
                vertices[np.array([2, 4, 7])],

                # top plane
                vertices[np.array([3, 7, 5])],
                vertices[np.array([3, 6, 7])],

                # front plane
                vertices[np.array([4, 6, 1])],
                vertices[np.array([4, 7, 6])],
            ]

    def plotSurface(self):
        a = np.array(self.get_facets())
        a = np.reshape(a, (3 * len(a), 3))
        data = plot_mesh(vertices=a, color="#464646")
        m = gl.GLMeshItem()
        m.setMeshData(**data)
        m.setGLOptions("opaque")
        return m







