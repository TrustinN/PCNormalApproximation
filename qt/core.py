import numpy as np

from samples.ball import Ball
from samples.ballAndBox import ballAndBox
from samples.box import Box
from samples.plane import Plane
from utils import PCtoSurface, visualizeNormals, visualizePC, visualizeProp


class Core:
    def __init__(self, console, display):
        self.console = console
        self.display = display
        self.toggle_functions = {}
        for keys in display.plotObjects.keys():
            self.toggle_functions[keys] = self.toggle_factory(keys)

        self.connect_pt()

        ## import project here, connect to display via plotObjects
        k = 15
        prop = Box(center=np.array([0, 0, 0]), x=1, y=1, z=1)
        pc = prop.sample(1000)
        s = PCtoSurface(pc=pc)
        s.computeTPs(k)
        s.computeRiemanianGraph(k)
        s.computeTraversalMST()
        s.computeMesh()
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
        self.display.loadItems(plotItems)

    def connect_pt(self):
        self.params = self.console.options.params
        for key in self.toggle_functions:
            self.params.child("display").child(key).sigTreeStateChanged.connect(
                self.toggle_functions[key]
            )

    def toggle_factory(self, name):
        def toggle_func():
            on = self.params.child("display").child(name).value()
            if on:
                self.display.show(name)
            else:
                self.display.hide(name)

        return toggle_func
