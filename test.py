import numpy as np
import pyqtgraph as pg
from samples.ball import Ball
from utils import makeView
from utils import visualizeProp, visualizePC, visualizeNormals
from utils import tangentPlanes
from utils import getCentroid, covMatrix

from rtrees.rstar_tree import RTree
from utils import TargetVertex

np.random.seed(100)

app = pg.mkQApp("pcNormals")
view = makeView()

prop = Ball(radius=5, center=np.array([0, 0, 0]))
pc = prop.sample(10, pitchRange=(np.pi / 6, np.pi / 7), yawRange=(np.pi / 6, np.pi / 7))

tree = RTree(M=10, dim=3)
for p in pc:
    tree.Insert(TargetVertex(value=p))

tP = tangentPlanes(pc=pc, tree=tree, numNeighbors=15)
c = getCentroid(pc)
visualizePC(view, [c], color="#ff0000")

clines = np.array([[0.05369291, 0.01402382, -0.02734346],
                   [0.0773394, -0.02912107, -0.02811782],
                   [0.10685366, -0.0134768, -0.04666212],
                   [0.10239754, -0.03189129, -0.03986704],
                   [0.07141908, 0.09785348, -0.05833208],
                   [0.05125591, 0.12466261, -0.05590197],
                   [-0.06654223,  0.0303623, 0.0250473],
                   [-0.09718455, -0.08905347, 0.06762068],
                   [-0.12181558, -0.05570042, 0.07076973],
                   [-0.17741614, -0.04765916, 0.09278679],])

cpoints = np.repeat(np.array([c]), [len(clines)], axis=0)
# print(cpoints)
cov = covMatrix(pc)
visualizeNormals(view, cpoints, clines)

normals = [p.normal for p in tP]
visualizePC(view, pc)
# visualizeNormals(view, pc, normals)
visualizeProp(view, prop)


lines = np.array([[-0.81381958, -0.41203117, 0.40979019],
                  [-0.34836516, 0.91032993, 0.22347514],
                  [0.465123, -0.03911182, 0.88438163],
                  ])

# lines = np.array([lines[2], lines[2], lines[2]])
visualizeNormals(view, np.array([c, c, c]), lines)


pg.exec()








