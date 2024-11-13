import pyqtgraph as pg
import pyqtgraph.opengl as gl

pg.mkQApp("pcNormals")
view = gl.GLViewWidget()
g = gl.GLGridItem()
print(view.cameraParams())
view.setCameraPosition(view.cameraParams())
