import math
import timeit
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any

import numpy as np
import pyqtgraph.opengl as gl
from colorutils import Color

from lookup import EdgeVertexIndices, TriangleTable, VertexPosition
from rtrees.plot import plot_mesh
from rtrees.rstar_tree import RTree
from rtrees.rtree_utils import Cube, IndexRecord


class OrientedTP:
    """
    Tangent plane described by a normal vector from center point
    """

    def __init__(self, center, normal):
        self.center = center
        self.normal = normal

    def offset(tp1, tp2):
        """
        Metric for tangency between two tangent planes.
        Assigns a high value to normals that are tangent
        and a lower value to normals that are collinear

        Args:
            tp1 (OrientedTP): First tangent plane.
            tp2 (OrientedTP): Second tangent plane.

        Returns:
            float: The tangency metric
        """

        return 1 - abs(np.dot(tp1.normal, tp2.normal))


def getCentroid(points: np.ndarray):
    """
    Computes center of mass of a collection of points

    Args:
        points (List[np.ndarray]): Collection of points

    Returns:
        np.ndarray: Center of mass
    """

    return sum(points) / len(points)


def getTangentPlane(points):
    """
    PCA computation via SVD

    Args:
        points (List[np.ndarray]): List of points

    Returns:
        OrientedTP: Oriented tangent plane
    """

    centroid = getCentroid(points)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1, :]  # Eigenvector with smallest eigenvalue

    return OrientedTP(center=getCentroid(points), normal=normal)


def tangentPlanes(pc, tree, numNeighbors=15):
    """
    Approximates tangent planes at each point in the point cloud.

    Args:
        pc (List[np.ndarray]): Point cloud points
        tree (RTree): Data structure containing points
        numNeighbors (int): Number of nearest neighbors to use

    Returns:
        List[OrientedTP]: List of tangent planes
    """

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
    points.setGLOptions("translucent")
    return points


def plotNormals(points, normals):
    a1 = np.array(points)
    a2 = np.array(normals)
    a2 = a1 + a2

    c = np.empty((len(a1) + len(a2), 3), dtype=a2.dtype)
    c[0::2] = a1
    c[1::2] = a2

    lines = gl.GLLinePlotItem(pos=c, mode="lines")
    lines.setGLOptions("opaque")

    return lines


def visualizeProp(prop):
    propPlot = prop.plotSurface()
    return propPlot


def visualizePC(pc, color="#464141"):
    pcPlot = plotPoints(pc, color)
    return pcPlot


def visualizeNormals(pc, normals):
    normalsPlot = plotNormals(points=pc, normals=normals)
    return normalsPlot


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


class Graph:
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

        lines = gl.GLLinePlotItem(pos=np.array(lines), mode="lines", color=(1, 0, 0, 1))
        lines.setGLOptions("translucent")
        return lines


class EMST:

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
        visited = np.zeros(len(self.vertices))
        visited[0] = 1

        seed = self.vertices[0]
        tree.Delete(seed)
        nn = tree.NearestNeighbor(seed)[0]

        pq = PriorityQueue()
        pq.put(PrioritizedItem(-math.inf, (seed, nn)))
        edges = 0

        while True:
            # pq item of form (p1, p2) where p1 is in current MST
            # p2 is vertex to be added

            v1, v2 = pq.get().item
            if not visited[v2.id]:
                v1.edgeTo(v2)
                v2.edgeTo(v1)
                edges += 1
                visited[v2.id] = 1
                tree.Delete(v2)

                if edges == len(self.vertices) - 1:
                    break

                v1NN = tree.NearestNeighbor(v1)[0]
                v2NN = tree.NearestNeighbor(v2)[0]
                v1Dist = np.linalg.norm(v1.tuple_identifier - v1NN.tuple_identifier)
                v2Dist = np.linalg.norm(v2.tuple_identifier - v2NN.tuple_identifier)
                pq.put(PrioritizedItem(v1Dist, (v1, v1NN)))
                pq.put(PrioritizedItem(v2Dist, (v2, v2NN)))

    def visualizeEdges(self):
        lines = []
        for v in self.vertices:
            for n in v.neighbors:
                v1 = v.tuple_identifier
                lines.append(v1)

                v2 = n.tuple_identifier
                lines.append(v2)

        lines = gl.GLLinePlotItem(pos=np.array(lines), mode="lines", color=(1, 0, 0, 1))
        lines.setGLOptions("translucent")
        return lines


class RiemanianGraph(EMST):

    def __init__(self, nodes, tree, k=15):
        self.vertices = nodes
        self.longestEdge = -math.inf

        start = timeit.default_timer()
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
        end = timeit.default_timer()
        print("Riemanian Graph took", end - start, "seconds")

        start = timeit.default_timer()
        super().__init__(self, tree)
        end = timeit.default_timer()
        print("EMST took", end - start, "seconds")

    def visualizeEdges(self):
        lines = []
        for v in self.vertices:
            for n in v.neighbors:
                v1 = v.tuple_identifier
                lines.append(v1)

                v2 = n.tuple_identifier
                lines.append(v2)

        lines = gl.GLLinePlotItem(pos=np.array(lines), mode="lines", color=(1, 0, 0, 1))
        lines.setGLOptions("translucent")
        return lines

    def getMST(self, weight):
        copy = []
        adjList = [[] for i in range(len(self.vertices))]
        visited = np.zeros(len(self.vertices))
        visited[0] = 1
        for v in self.vertices:
            n = RiemanianGraph.Node(v.tuple_identifier, v.item, v.id)
            copy.append(n)

        edges = 0
        pq = PriorityQueue()
        seed = self.vertices[0]
        for v in seed.neighbors:
            pq.put(PrioritizedItem(weight(seed, v), (seed, v)))

        while True:
            # pq item of form (p1, p2) where p1 is in current MST
            # p2 is vertex to be added

            v1, v2 = pq.get().item
            if not visited[v2.id]:
                adjList[v1.id].append(v2.id)
                adjList[v2.id].append(v1.id)
                visited[v2.id] = 1
                edges += 1

                if edges == len(self.vertices) - 1:
                    break

                for v in v1.neighbors:
                    if not visited[v.id]:
                        pq.put(PrioritizedItem(weight(v1, v), (v1, v)))
                for v in v2.neighbors:
                    if not visited[v.id]:
                        pq.put(PrioritizedItem(weight(v2, v), (v2, v)))

        for v1 in range(len(adjList)):
            row = adjList[v1]
            for v2 in row:
                copy[v1].neighbors.append(copy[v2])

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

    def dist(point, tp):
        return np.dot(point - tp.center, tp.normal)

    def sdf(x):
        tpNear = tpTree.NearestNeighbor(IndexRecord(bound=None, tuple_identifier=x))[
            0
        ].item
        projNorm = dist(x, tpNear)
        projX = x - projNorm * tpNear.normal

        zNN = pcTree.NearestNeighbor(IndexRecord(bound=None, tuple_identifier=projX))[
            0
        ].tuple_identifier
        # if np.linalg.norm(projX - zNN) < delta + ro:
        return projNorm
        #
        # else:
        #     localTP = tpTree.NearestNeighbor(IndexRecord(bound=None, tuple_identifier=x), k=20)
        #     estCenter = sum([v.item.center for v in localTP]) / len(localTP)
        #
        #     distances = np.array([np.linalg.norm(estCenter - v.item.center) for v in localTP])
        #     totalDist = sum(distances)
        #
        #     weights = distances / totalDist
        #     tangents = np.array([v.item.normal for v in localTP])
        #     estTangent = np.dot(weights, tangents)
        #     estNormal = estTangent / np.linalg.norm(estTangent)
        #
        #     projNorm = dist(x, OrientedTP(center=estCenter, normal=estNormal))
        #     projX = x - projNorm * estNormal
        #
        #     return projNorm

    return sdf


class PCtoSurface:
    def __init__(self, pc, k=15):
        self.pc = pc

        self.pcTree = RTree(M=10, dim=3)
        for p in pc:
            self.pcTree.Insert(TargetVertex(value=p))

    def computeTPs(self, k=15):
        self.tP = tangentPlanes(pc=self.pc, tree=self.pcTree, numNeighbors=k)
        self.tPNodes = [
            EMST.Node(self.tP[i].center, self.tP[i], i) for i in range(len(self.tP))
        ]
        self.centers = [p.center for p in self.tP]

        self.tpTree = RTree(M=10, dim=3)
        for tp in self.tPNodes:
            self.tpTree.Insert(tp)

    def computeRiemanianGraph(self, k=15):
        self.rG = RiemanianGraph(self.tPNodes, self.tpTree, k=k)

    def computeTraversalMST(self):
        start = timeit.default_timer()
        self.mst = self.rG.getMST(RiemanianGraph.Node.compareItems(OrientedTP.offset))
        end = timeit.default_timer()
        print("MST took", end - start, "seconds")

    def computeMesh(self):
        fixOrientations(self.mst)
        self.normals = [p.normal for p in self.tP]

        self.tpTree = RTree(M=10, dim=3)
        for tp in self.tPNodes:
            self.tpTree.Insert(tp)

        self.boundingBox = self.tpTree.root.covering
        self.sdf = getSDF(
            pc=self.pc, tp=self.tPNodes, delta=0, ro=self.rG.longestEdge / 2
        )

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

    def visualizePoints(self, color="#464141"):
        return visualizePC(self.pc, color=color)

    def visualizeTPCenters(self, color="#464141"):
        return visualizePC(self.centers, color=color)

    def visualizeTPNormals(self):
        return visualizeNormals(self.centers, self.normals)

    def visualizeRiemanianGraph(self):
        return self.rG.visualizeEdges()

    def visualizeTraversalMST(self):
        return self.mst.visualizeEdges()

    def visualizeSurface(self):
        return self.mesh


def MarchingCubes(surface, length):

    def calcCubeIndex(x, y, z, sdfMesh):
        sdfs = sdfMesh[
            x + VertexPosition[:, 0], y + VertexPosition[:, 1], z + VertexPosition[:, 2]
        ]
        signs = sdfs >= 0
        return signs.dot(2 ** np.arange(8))

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

            v00, v01 = EdgeVertexIndices[edges[idx]]
            v10, v11 = EdgeVertexIndices[edges[idx + 1]]
            v20, v21 = EdgeVertexIndices[edges[idx + 2]]

            v0 = interpolate(
                pos + length * VertexPosition[v00], pos + length * VertexPosition[v01]
            )
            v1 = interpolate(
                pos + length * VertexPosition[v10], pos + length * VertexPosition[v11]
            )
            v2 = interpolate(
                pos + length * VertexPosition[v20], pos + length * VertexPosition[v21]
            )

            triangles.append(v0)
            triangles.append(v1)
            triangles.append(v2)

    def vectorizedSDFs(minVals, maxVals, length):
        coords = np.stack(
            np.meshgrid(
                *[np.arange(minVals[i], maxVals[i] + length, length) for i in range(3)]
            ),
            axis=-1,
        )
        coords = np.moveaxis(coords, 0, 1)
        result = np.apply_along_axis(surface.getSDF(), axis=-1, arr=coords)
        return result

    def march():

        for i in range(divisions[0]):
            for j in range(divisions[1]):
                for k in range(divisions[2]):

                    currCorner = minVals + np.array(
                        [i * length, j * length, k * length]
                    )
                    cubeIndex = calcCubeIndex(i, j, k, sdfGrid3D)

                    edges = TriangleTable[cubeIndex]
                    parseEdges(edges, currCorner)

    sdf = surface.getSDF()
    bb = surface.boundingBox
    bb = Cube(
        [
            bb.min_x - length / 2,
            bb.max_x,
            bb.min_y - length / 2,
            bb.max_y,
            bb.min_z - length / 2,
            bb.max_z,
        ]
    )

    minVals = np.array([bb.min_x, bb.min_y, bb.min_z])
    maxVals = np.array([bb.max_x, bb.max_y, bb.max_z])
    diff = maxVals - minVals

    divisions = np.ceil(diff / length)
    divisions = divisions.astype(np.int64)
    sdfGrid3D = vectorizedSDFs(minVals, minVals + divisions * length, length)
    triangles = []
    march()
    m = gl.GLMeshItem()
    m.setMeshData(**plot_mesh(np.array(triangles), color="#6ecd00"))

    return m


class Voxel:
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
            corner + x + y + z,
        ]
