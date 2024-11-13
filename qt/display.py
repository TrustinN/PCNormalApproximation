import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QVector3D
from pyqtgraph.Qt import QtWidgets


class Display(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        pg.mkQApp("pcNormals")
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        view = gl.GLViewWidget()
        g = gl.GLGridItem()
        view.addItem(g)
        view.show()
        self.view = view
        view.cameraPosition()
        self.layout.addWidget(self.view)
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateCamera)
        self.timer.start(16)

        self.plotObjects = {
            "prop": None,
            "point cloud": None,
            "tangent centers": None,
            "normals": None,
            "riemanian graph": None,
            "emst": None,
            "traversal order": None,
            "mesh": None,
        }

    def loadItems(self, items):
        for key in items:
            v = items[key]
            self.plotObjects[key] = v
            if v:
                self.view.addItem(v)
                v.hide()

    def unloadItems(self):
        for key in self.plotObjects:
            if self.plotObjects[key]:
                self.view.removeItem(self.plotObjects[key])
                item = self.plotObjects[key]
                self.plotObjects[key] = None
                del item

    def hide(self, name):
        if self.plotObjects[name]:
            return self.plotObjects[name].hide()

    def show(self, name):
        if self.plotObjects[name]:
            return self.plotObjects[name].show()

    def updateCamera(self):
        params = self.view.cameraParams()
        params["azimuth"] = (params["azimuth"] + 1) % 360
        self.view.setCameraPosition(azimuth=params["azimuth"])
