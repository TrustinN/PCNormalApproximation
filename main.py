import numpy as np
import pyqtgraph as pg
from samples.ball import Ball
from samples.box import Box
from samples.plane import Plane

from utils import PCtoSurface
from utils import makeView
from utils import MarchingCubes

np.random.seed(100)

app = pg.mkQApp("pcNormals")
view = makeView()


# prop = Plane(radius=5, center=np.array([0, 0, 0]))
# pc = prop.sampleNoise(1000)

prop = Ball(radius=1, center=np.array([0, 0, 0]))
pc = prop.sample(1000)

# prop = Box(center=np.array([0, 0, 0]), x=3, y=5, z=2)
# pc = prop.sample(3000)

s = PCtoSurface(pc=pc)

# s.visualizeTP(view)
# s.visualizeRiemanianGraph(view)
# visualizeProp(view, prop)
s.visualizeSurface(view)

pg.exec()

















