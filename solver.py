import timeit

import numpy as np
from samples.ball import Ball
from samples.ballAndBox import ballAndBox
from samples.box import Box
from samples.plane import Plane

from utils import PCtoSurface, visualizeProp


class PCSolver:

    def __init__(self):
        self.reload()
        self.profile = {}

    def reload(self):
        self.props = {
            "plane": Plane(radius=3, center=np.array([0, 0, 0])),
            "ball": Ball(center=np.array([0, 0, 0]), radius=3),
            "box": Box(center=np.array([0, 0, 0]), x=2, y=3, z=2),
            "ballAndBox": ballAndBox(center=np.array([0, 0, 0]), radius=3),
        }

    def setProp(self, name):
        self.prop = self.props[name]

    def solve(self, samples):

        k = 15
        prop = self.prop
        pc = prop.sample(samples)
        s = PCtoSurface(pc=pc)

        start = timeit.default_timer()
        s.computeTPs(k)
        end = timeit.default_timer()
        self.profile["Tangent Plane Calc"] = end - start

        start = timeit.default_timer()
        s.computeRiemanianGraph(k)
        end = timeit.default_timer()
        self.profile["Riemanian Graph"] = end - start

        start = timeit.default_timer()
        s.computeTraversalMST()
        end = timeit.default_timer()
        self.profile["Traversal MST"] = end - start

        start = timeit.default_timer()
        s.computeMesh()
        end = timeit.default_timer()
        self.profile["Marching Cubes"] = end - start

        plotItems = {
            "prop": visualizeProp(prop),
            "point cloud": s.visualizePoints(),
            "tangent centers": s.visualizeTPCenters(),
            "normals": s.visualizeTPNormals(),
            "riemanian graph": s.visualizeRiemanianGraph(),
            "emst": None,
            "traversal order": s.visualizeTraversalMST(),
            "mesh": s.visualizeSurface(),
        }
        return plotItems
