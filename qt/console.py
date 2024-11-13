import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


class Console(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.setFixedSize(300, 900)

        self.options = Options()
        self.layout.addWidget(self.options.pt)


class Options:
    def __init__(self):

        self.opts = [
            dict(
                name="Props",
                type="list",
                limits=[
                    "ball",
                    "box",
                ],
                value="ball",
                children=[
                    dict(name="num_sample", type="float", value=1000),
                ],
            ),
            dict(
                name="display",
                type="list",
                children=[
                    dict(name="prop", type="bool", value=False),
                    dict(name="point cloud", type="bool", value=False),
                    dict(name="tangent centers", type="bool", value=False),
                    dict(name="normals", type="bool", value=False),
                    dict(name="riemanian graph", type="bool", value=False),
                    dict(name="emst", type="bool", value=False),
                    dict(name="traversal order", type="bool", value=False),
                    dict(name="mesh", type="bool", value=False),
                ],
            ),
            dict(name="run", type="action"),
        ]

        self.params = pg.parametertree.Parameter.create(
            name="Parameters", type="group", children=self.opts
        )

        self.pt = pg.parametertree.ParameterTree()
        self.pt.setParameters(self.params)
