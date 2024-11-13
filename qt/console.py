import pyqtgraph as pg
from PyQt6.QtCore import Qt
from pyqtgraph.Qt import QtWidgets


class ConsoleData:

    def __init__(self, keys):
        self.labels = {}
        self.data = {}
        for k in keys:
            self.labels[k] = QtWidgets.QLabel()
            self.labels[k].setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.labels[k].setText(f"{k}:")

            self.data[k] = QtWidgets.QLabel()
            self.data[k].setAlignment(Qt.AlignmentFlag.AlignRight)

        self.window = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()
        self.window.setLayout(self.layout)

        self.data_left = QtWidgets.QWidget()
        self.data_left_layout = QtWidgets.QVBoxLayout()
        self.data_left.setLayout(self.data_left_layout)

        self.data_right = QtWidgets.QWidget()
        self.data_right_layout = QtWidgets.QVBoxLayout()
        self.data_right.setLayout(self.data_right_layout)

        for k in keys:
            self.data_left_layout.addWidget(self.labels[k])
            self.data_right_layout.addWidget(self.data[k])

        self.layout.addWidget(self.data_left)
        self.layout.addWidget(self.data_right)


class Console(QtWidgets.QWidget):
    def __init__(self, dataLabels):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.setFixedSize(300, 900)

        self.options = Options()
        self.data = ConsoleData(dataLabels)
        self.layout.addWidget(self.options.pt)
        self.layout.addWidget(self.data.window)

    def updateData(self, data):
        for key in data:
            self.data.data[key].setText(
                str(int(10000 * data[key]) / 10000) + " seconds"
            )


class Options:
    def __init__(self):

        self.opts = [
            dict(
                name="Props",
                type="list",
                limits=["plane", "ball", "box", "ballAndBox"],
                value="ball",
                children=[
                    dict(name="samples", type="int", value=1000),
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
