import numpy as np
import pyqtgraph as pg
from samples.ball import Ball
from samples.box import Box
from samples.plane import Plane

from rtrees.rstar_tree import RTree

from utils import TargetVertex, Graph, EMST, RiemanianGraph, OrientedTP
from utils import makeView
from utils import tangentPlanes
from utils import visualizeProp, visualizePC, visualizeNormals
from utils import fixOrientations

np.random.seed(100)

app = pg.mkQApp("pcNormals")
view = makeView()


# prop = Plane(radius=5, center=np.array([0, 0, 0]))
# pc = prop.sampleNoise(2000)

prop = Ball(radius=1, center=np.array([0, 0, 0]))
pc = prop.sample(1000)

# prop = Box(center=np.array([0, 0, 0]), x=1, y=1, z=1)
# pc = prop.sample(3000)

tree = RTree(M=10, dim=3)
for p in pc:
    tree.Insert(TargetVertex(value=p))

tP = tangentPlanes(pc=pc, tree=tree, numNeighbors=10)
tPNodes = [EMST.Node(tP[i].center, tP[i], i) for i in range(len(tP))]
centers = [p.center for p in tP]

tree2 = RTree(M=10, dim=3)
for tp in tPNodes:
    tree2.Insert(tp)

# emst = EMST(Graph(tPNodes), tree2)
# view.addItem(emst.visualizeEdges())

rG = RiemanianGraph(tPNodes, tree2, k=10)
mst = rG.getMST(RiemanianGraph.Node.compareItems(OrientedTP.offset))
view.addItem(mst.visualizeEdges())

fixOrientations(mst)
normals = [p.normal for p in tP]

visualizePC(view, centers)
visualizeNormals(view, centers, normals)
visualizeProp(view, prop)

pg.exec()














