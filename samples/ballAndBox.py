import numpy as np


class ballAndBox:
    def __init__(self, center, radius):
        self.radius = radius
        self.center = center

        self.domeSA = 6 * 2 * np.pi * self.radius**2
        self.cubeSA = 6 * (4 - np.pi) * self.radius**2
        self.totalSA = self.domeSA + self.cubeSA

        self.lengths = [2 * self.radius for i in range(3)]
        self.margin = sum(self.lengths)
        self.ranges = np.repeat(np.array([[-self.radius, self.radius]]), [3], axis=0)

    def sample(self, num):
        samples = []
        while num > 0:
            c = np.random.choice(
                [0, 1],
                size=1,
                p=[self.domeSA / self.totalSA, self.cubeSA / self.totalSA],
            )
            if c == 0:
                z = 2 * self.radius * np.random.random_sample() - self.radius
                theta = 2 * np.pi * np.random.random_sample()
                a = np.sqrt(self.radius**2 - z**2)
                x = np.cos(theta) * a
                y = np.sin(theta) * a
                if z < 0:
                    z -= self.radius

                else:
                    z += self.radius

                samples.append(np.random.permutation([x, y, z]) + self.center)
                num -= 1

            else:
                while True:
                    c = np.random.choice(
                        [0, 1, 2],
                        size=2,
                        replace=False,
                        p=[ln / self.margin for ln in self.lengths],
                    )
                    c_comp = 3 - (c[0] + c[1])

                    s = self.ranges[c]
                    s_comp = self.ranges[c_comp]

                    p_rand = np.array([0.0, 0.0, 0.0])
                    p_rand[c] = np.array(
                        [(r[1] - r[0]) * np.random.random_sample() + r[0] for r in s]
                    )
                    p_rand[c_comp] = s_comp[np.random.randint(2, size=1)]

                    planeCenter = np.array([0.0, 0.0, 0.0])
                    planeCenter[c_comp] = p_rand[c_comp]

                    if np.linalg.norm(p_rand - planeCenter) >= self.radius:
                        samples.append(p_rand + self.center)
                        num -= 1
                        break

        return samples
