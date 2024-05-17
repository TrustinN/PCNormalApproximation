import math
import numpy as np
import pyqtgraph.opengl as gl
from operator import itemgetter
from colorutils import Color
from rtrees.rstar_tree import RTree
from rtrees.rtree_utils import IndexRecord
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any


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

        for v in self.vertices:
            kNN = tree.NearestNeighbor(v, k=k)
            for p in kNN:
                if p != v:
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











