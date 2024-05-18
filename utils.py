import math
import numpy as np
import pyqtgraph.opengl as gl
from colorutils import Color
from rtrees.rstar_tree import RTree
from rtrees.rtree_utils import IndexRecord
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
from lookup import EdgeVertexIndices, TriangleTable, VertexPosition
from rtrees.plot import plot_mesh
from rtrees.rtree_utils import Cube


class OrientedTP():
    def __init__(self, center, normal):
        self.center = center
        self.normal = normal

    def offset(tp1, tp2):
        return 1 - abs(np.dot(tp1.normal, tp2.normal))


def getCentroid(points):
    return sum(points) / len(points)


def svdCalc(points):

    centroid = getCentroid(points)
    svd = np.linalg.svd(points - centroid)
    return svd


def getTangentPlane(points):

    svd = svdCalc(points)
    normal = svd[2][-1, :]

    return OrientedTP(center=getCentroid(points), normal=normal)


def tangentPlanes(pc, tree, numNeighbors=15):

    tP = []
    for p in pc:
        nn = tree.NearestNeighbor(entry=TargetVertex(value=p), k=numNeighbors)
        nn = [r.tuple_identifier for r in nn]
        tP.append(getTangentPlane(nn))

    return tP


def makeView():
    view = gl.GLViewWidget()

    g = gl.GLGridItem()
    view.addItem(g)
    view.show()

    return view


def plotPoints(points, color):
    c = np.array([Color(web=color).rgb])
    cm = np.repeat(c, len(points), axis=0)
    points = gl.GLScatterPlotItem(pos=np.array(points), color=cm, size=5)
    points.setGLOptions('translucent')
    return points


def plotNormals(points, normals):
    a1 = np.array(points)
    a2 = np.array(normals)
    a2 = a1 + a2

    c = np.empty((len(a1) + len(a2), 3), dtype=a2.dtype)
    c[0::2] = a1
    c[1::2] = a2

    lines = gl.GLLinePlotItem(pos=c, mode='lines')
    lines.setGLOptions('opaque')

    return lines


def visualizeProp(view, prop):
    propPlot = prop.plotSurface()
    view.addItem(propPlot)


def visualizePC(view, pc, color="#464141"):
    pcPlot = plotPoints(pc, color)
    view.addItem(pcPlot)


def visualizeNormals(view, pc, normals):

    normalsPlot = plotNormals(points=pc, normals=normals)
    view.addItem(normalsPlot)


class TargetVertex(IndexRecord):
    def __init__(self, value):
        super().__init__(bound=None, tuple_identifier=value)

    def __eq__(self, other):
        return np.array_equal(self.tuple_identifier, other.tuple_identifier)

    def __neq__(self, other):
        return not self.__eq__(other)


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


class Graph():
    def __init__(self, vertices):
        self.vertices = vertices

    def visualizeEdges(self):
        lines = []
        for v in self.vertices:
            for n in v.neighbors:
                v1 = v.tuple_identifier
                lines.append(v1)

                v2 = n.tuple_identifier
                lines.append(v2)

        lines = gl.GLLinePlotItem(pos=np.array(lines), mode='lines', color=(1, 0, 0, 1))
        lines.setGLOptions('translucent')
        return lines


class EMST():

    class Node(TargetVertex):
        def __init__(self, value, item, id):
            super().__init__(value)
            self.neighbors = []
            self.item = item
            self.id = id

        def __eq__(self, other):
            return super().__eq__(other) and self.id == other.id

        def __neq__(self, other):
            return not self.__eq__(other)

        def compareValues(f):
            def c(n1, n2):
                return f(n1.tuple_identifier, n2.tuple_identifier)

            return c

        def compareItems(f):
            def c(n1, n2):
                return f(n1.item, n2.item)

            return c

        def edgeTo(self, other):
            self.neighbors.append(other)

    def __init__(self, graph, tree):

        self.vertices = graph.vertices
        nn = [None] * len(self.vertices)
        distToNN = [None] * len(self.vertices)
        visited = np.zeros(len(self.vertices))
        visited[0] = 1

        seed = self.vertices[0]
        tree.Delete(seed)

        pq = PriorityQueue()
        pq.put(PrioritizedItem(-math.inf, seed))
        edges = 0

        while edges < len(self.vertices) - 1:
            while True:
                v = pq.queue[0]
                if v.priority == -math.inf:
                    v = pq.get().item
                    n = tree.NearestNeighbor(v)[0]

                    nn[v.id] = n
                    distToNN[v.id] = np.linalg.norm(v.tuple_identifier - n.tuple_identifier)

                    e = PrioritizedItem(distToNN[v.id], v)
                    pq.put(e)

                else:
                    break

            closest = pq.get().item
            vNN = nn[closest.id]

            if not visited[vNN.id]:

                visited[vNN.id] = 1
                tree.Delete(vNN)

                closest.edgeTo(vNN)
                vNN.edgeTo(closest)
                edges += 1
                pq.put(PrioritizedItem(-math.inf, closest))
                pq.put(PrioritizedItem(-math.inf, vNN))

            else:
                nn[closest.id] = None
                distToNN[closest.id] = None
                pq.put(PrioritizedItem(-math.inf, closest))

    def visualizeEdges(self):
        lines = []
        for v in self.vertices:
            for n in v.neighbors:
                v1 = v.tuple_identifier
                lines.append(v1)

                v2 = n.tuple_identifier
                lines.append(v2)

        lines = gl.GLLinePlotItem(pos=np.array(lines), mode='lines', color=(1, 0, 0, 1))
        lines.setGLOptions('translucent')
        return lines


class RiemanianGraph(EMST):

    def __init__(self, nodes, tree, k=15):
        self.vertices = nodes
        self.longestEdge = -math.inf

        for v in self.vertices:
            kNN = tree.NearestNeighbor(v, k=k)
            closest = False
            for i in range(len(kNN)):
                p = kNN[i]
                if p != v:
                    if not closest:
                        closest = True
                        dist = np.linalg.norm(p.tuple_identifier - v.tuple_identifier)
                        if dist > self.longestEdge:
                            self.longestEdge = dist
                    v.neighbors.append(p)
                    p.neighbors.append(v)

        super().__init__(self, tree)

    def visualizeEdges(self):
        lines = []
        for v in self.vertices:
            for n in v.neighbors:
                v1 = v.tuple_identifier
                lines.append(v1)

                v2 = n.tuple_identifier
                lines.append(v2)

        lines = gl.GLLinePlotItem(pos=np.array(lines), mode='lines', color=(1, 0, 0, 1))
        lines.setGLOptions('translucent')
        return lines

    def getMST(self, weight):
        copy = []
        visited = np.zeros(len(self.vertices))
        edgeTo = [None] * len(self.vertices)
        costs = [math.inf] * len(self.vertices)

        for v in self.vertices:
            n = RiemanianGraph.Node(v.tuple_identifier, v.item, v.id)
            copy.append(n)

        edges = 0
        pq = PriorityQueue()
        pq.put(PrioritizedItem(0, self.vertices[0]))

        while edges < len(copy) - 1:

            v = pq.get().item
            if not visited[v.id]:
                visited[v.id] = 1

                if edgeTo[v.id]:

                    newV = edgeTo[v.id]
                    newC = copy[newV.id]

                    copy[v.id].neighbors.append(newC)
                    newC.neighbors.append(copy[v.id])
                    edges += 1

                    for n in newV.neighbors:
                        if not visited[n.id]:
                            cost = weight(newV, n)
                            if cost < costs[n.id]:
                                edgeTo[n.id] = newV
                                costs[n.id] = cost

                            e = PrioritizedItem(costs[n.id], n)
                            pq.put(e)

                for n in v.neighbors:
                    if not visited[n.id]:
                        cost = weight(v, n)
                        if cost < costs[n.id]:
                            edgeTo[n.id] = v
                            costs[n.id] = cost

                        e = PrioritizedItem(costs[n.id], n)
                        pq.put(e)

        return Graph(copy)


def fixOrientations(graph):
    visited = np.zeros(len(graph.vertices))
    start = max(graph.vertices, key=lambda v: v.tuple_identifier[2])

    if np.dot(start.item.normal, np.array([0, 0, 1])) < 0:
        start.item.normal = -start.item.normal

    # makes tangentPlane2 have the same normal vec
    # orientation as tangentPlane1
    def align(tp1, tp2):
        if np.dot(tp1.normal, tp2.normal) < 0:
            tp2.normal = -tp2.normal

    def DFS(v):
        visited[v.id] = 1
        for n in v.neighbors:
            if not visited[n.id]:
                align(v.item, n.item)
                DFS(n)

    DFS(start)


def getSDF(pc, tp, delta, ro):

    pcTree = RTree(M=10, dim=3)
    tpTree = RTree(M=10, dim=3)

    for p in pc:
        pcTree.Insert(IndexRecord(bound=None, tuple_identifier=p))

    for p in tp:
        tpTree.Insert(p)

    def sdf(x):
        tpNear = tpTree.NearestNeighbor(IndexRecord(bound=None, tuple_identifier=x))[0].item
        projNorm = np.dot(x - tpNear.center, tpNear.normal)
        projX = x - projNorm * tpNear.normal

        zNN = pcTree.NearestNeighbor(IndexRecord(bound=None, tuple_identifier=projX))[0].tuple_identifier
        # if np.linalg.norm(projX - zNN) < delta + ro:
        return projNorm
        #
        # else:
        #     return -math.inf

    return sdf


class PCtoSurface():
    def __init__(self, pc, k=15):
        self.pc = pc

        pcTree = RTree(M=10, dim=3)
        for p in pc:
            pcTree.Insert(TargetVertex(value=p))

        tP = tangentPlanes(pc=self.pc, tree=pcTree, numNeighbors=k)
        self.tPNodes = [EMST.Node(tP[i].center, tP[i], i) for i in range(len(tP))]
        self.centers = [p.center for p in tP]

        tpTree = RTree(M=10, dim=3)
        for tp in self.tPNodes:
            tpTree.Insert(tp)

        self.rG = RiemanianGraph(self.tPNodes, tpTree, k=k)
        self.mst = self.rG.getMST(RiemanianGraph.Node.compareItems(OrientedTP.offset))

        fixOrientations(self.mst)
        self.normals = [p.normal for p in tP]

        tpTree = RTree(M=10, dim=3)
        for tp in self.tPNodes:
            tpTree.Insert(tp)

        self.boundingBox = tpTree.root.covering
        self.sdf = getSDF(pc=self.pc, tp=self.tPNodes, delta=0, ro=self.rG.longestEdge / 2)

        self.mesh = MarchingCubes(self, 1.2 * self.rG.longestEdge)

    def getPoints(self):
        return self.pc

    def getTPCenters(self):
        return self.centers

    def getTPNormals(self):
        return self.normals

    def getRiemanianGraph(self):
        return self.rG

    def getTraversalMST(self):
        return self.mst

    def getSDF(self):
        return self.sdf

    def visualizePoints(self, view, color="#464141"):
        visualizePC(view, self.pc, color=color)

    def visualizeTPCenters(self, view, color="#464141"):
        visualizePC(view, self.centers, color=color)

    def visualizeTPNormals(self, view):
        visualizeNormals(view, self.centers, self.normals)

    def visualizeTP(self, view):
        self.visualizeTPCenters(view)
        self.visualizeTPNormals(view)

    def visualizeRiemanianGraph(self, view):
        view.addItem(self.rG.visualizeEdges())

    def visualizeTraversalMST(self, view):
        view.addItem(self.mst.visualizeEdges())

    def visualizeSurface(self, view):
        view.addItem(self.mesh)


def MarchingCubes(surface, length):

    def calcCubeIndex(cube):
        sdfs = np.array(list(map(surface.getSDF(), cube.vertices)))
        signs = sdfs >= 0
        return sum([2 ** i for i in range(len(signs)) if signs[i]])

    def interpolate(v1, v2):
        w1 = abs(sdf(v1))
        w2 = abs(sdf(v2))
        scale = w1 / (w1 + w2)
        v = scale * (v2 - v1) + v1
        return v

    def parseEdges(edges, pos):
        rep = (len(edges) - 1) // 3 + 1
        for i in range(rep):
            idx = 3 * i
            if edges[idx] == -1:
                break

            v00 = EdgeVertexIndices[edges[idx]][0]
            v01 = EdgeVertexIndices[edges[idx]][1]

            v10 = EdgeVertexIndices[edges[idx + 1]][0]
            v11 = EdgeVertexIndices[edges[idx + 1]][1]

            v20 = EdgeVertexIndices[edges[idx + 2]][0]
            v21 = EdgeVertexIndices[edges[idx + 2]][1]

            v0 = interpolate(pos + length * VertexPosition[v00], pos + length * VertexPosition[v01])
            v1 = interpolate(pos + length * VertexPosition[v10], pos + length * VertexPosition[v11])
            v2 = interpolate(pos + length * VertexPosition[v20], pos + length * VertexPosition[v21])

            triangles.append(v0)
            triangles.append(v1)
            triangles.append(v2)

    def march():
        for i in range(divisions[2]):
            zDisp = length * np.array([0, 0, i])

            for j in range(divisions[1]):
                yDisp = length * np.array([0, j, 0])

                for k in range(divisions[0]):
                    xDisp = length * np.array([k, 0, 0])

                    currCorner = minVals + xDisp + yDisp + zDisp
                    currCube = Voxel(currCorner, length)
                    cubeIndex = calcCubeIndex(currCube)

                    edges = TriangleTable[cubeIndex]
                    parseEdges(edges, currCorner)

    sdf = surface.getSDF()
    bb = surface.boundingBox
    bb = Cube([bb.min_x - length / 3, bb.max_x,
               bb.min_y - length / 3, bb.max_y,
               bb.min_z - length / 3, bb.max_z])

    minVals = np.array([bb.min_x, bb.min_y, bb.min_z])
    maxVals = np.array([bb.max_x, bb.max_y, bb.max_z])
    diff = maxVals - minVals

    divisions = np.ceil(diff / np.repeat(length, 3))
    divisions = divisions.astype(np.int64)
    triangles = []
    march()
    m = gl.GLMeshItem()
    m.setMeshData(**plot_mesh(np.array(triangles), color="#6ecd00"))

    return m


class Voxel():
    def __init__(self, corner, length):
        x = length * np.array([1, 0, 0])
        y = length * np.array([0, 1, 0])
        z = length * np.array([0, 0, 1])
        self.vertices = [
            corner,
            corner + x,
            corner + y,
            corner + x + y,
            corner + z,
            corner + x + z,
            corner + y + z,
            corner + x + y + z
        ]
























