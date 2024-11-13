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
    for i in range(len(pc)):
        ir = IndexRecord(bound=Cube(np.repeat(pc[i], 2)), tuple_identifier=i)
        nn = tree.NearestNeighbor(
            ir,
            k=numNeighbors,
        )
        nn = [pc[idx] for idx in nn]
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
    try:
        propPlot = prop.plotSurface()
        return propPlot
    except AttributeError:
        return None


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


class GraphUtils:
    def joinGraphs(adjList1, adjList2):
        assert len(adjList1) == len(adjList2)

        newAdjList = []
        for i in range(len(adjList1)):
            newAdjList.append(adjList1[i] | adjList2[i])

        return newAdjList

    def visualizeEdges(vertices, adjList):
        lines = []
        for vIdx in range(len(vertices)):
            v1 = vertices[vIdx]
            for nIdx in adjList[vIdx]:
                lines.append(v1)

                v2 = vertices[nIdx]
                lines.append(v2)

        lines = gl.GLLinePlotItem(pos=np.array(lines), mode="lines", color=(1, 0, 0, 1))
        lines.setGLOptions("translucent")
        return lines


class EMST:
    def __init__(self, nodes, tree):

        self.vertices = nodes
        self.adjList = [set() for i in range(len(self.vertices))]
        visited = np.zeros(len(self.vertices))
        visited[0] = 1

        seed = self.vertices[0]
        seedIR = IndexRecord(bound=Cube(np.repeat(seed, 2)), tuple_identifier=0)
        tree.Delete(seedIR)
        nn = tree.NearestNeighbor(seedIR)[0]

        pq = PriorityQueue()
        pq.put(PrioritizedItem(-math.inf, (0, nn)))
        edges = 0

        while True:
            # pq item of form (p1, p2) where p1 is in current MST
            # p2 is vertex to be added

            v1, v2 = pq.get().item
            v1Vertex = self.vertices[v1]
            v2Vertex = self.vertices[v2]
            v1IR = IndexRecord(bound=Cube(np.repeat(v1Vertex, 2)), tuple_identifier=v1)
            v2IR = IndexRecord(bound=Cube(np.repeat(v2Vertex, 2)), tuple_identifier=v2)
            if not visited[v2]:
                self.adjList[v1].add(v2)
                self.adjList[v2].add(v1)
                edges += 1
                visited[v2] = 1
                tree.Delete(v2IR)

                if edges == len(self.vertices) - 1:
                    break

                v1NN = tree.NearestNeighbor(v1IR)[0]
                v2NN = tree.NearestNeighbor(v2IR)[0]
                v1Dist = np.linalg.norm(v1Vertex - self.vertices[v1NN])
                v2Dist = np.linalg.norm(v2Vertex - self.vertices[v2NN])
                pq.put(PrioritizedItem(v1Dist, (v1, v1NN)))
                pq.put(PrioritizedItem(v2Dist, (v2, v2NN)))

    def visualizeEdges(self):
        return GraphUtils.visualizeEdges(self.vertices, self.adjList)


class RiemanianGraph:

    def __init__(self, nodes, tree, k=15):
        self.vertices = nodes
        self.longestEdge = -math.inf
        self.adjList = [set() for i in range(len(self.vertices))]

        for i in range(len(self.vertices)):
            v = self.vertices[i]
            kNN = tree.NearestNeighbor(
                IndexRecord(bound=Cube(np.repeat(v, 2)), tuple_identifier=i), k=k
            )
            self.longestEdge = max(
                np.linalg.norm(self.vertices[kNN[1]] - v),
                self.longestEdge,
            )
            for j in range(1, len(kNN)):
                p = kNN[j]
                self.adjList[p].add(i)
                self.adjList[i].add(p)

    def visualizeEdges(self):
        return GraphUtils.visualizeEdges(self.vertices, self.adjList)


def getMST(vertices, adjList, weight):
    returnAdjList = [set() for i in range(len(vertices))]
    visited = np.zeros(len(vertices))
    visited[0] = 1

    edges = 0
    pq = PriorityQueue()
    for v in adjList[0]:
        pair = (vertices[0], vertices[v])
        pq.put(PrioritizedItem(weight(*pair), (0, v)))

    while True:
        # pq item of form (p1, p2) where p1 is in current MST
        # p2 is vertex to be added

        v1, v2 = pq.get().item
        if not visited[v2]:
            returnAdjList[v1].add(v2)
            returnAdjList[v2].add(v1)
            visited[v2] = 1
            edges += 1

            if edges == len(vertices) - 1:
                break

            for v in adjList[v1]:
                if not visited[v]:
                    pq.put(PrioritizedItem(weight(vertices[v1], vertices[v]), (v1, v)))

            for v in adjList[v2]:
                if not visited[v]:
                    pq.put(PrioritizedItem(weight(vertices[v2], vertices[v]), (v2, v)))

    return returnAdjList


def fixOrientations(tangentPlanes, adjList):
    visited = np.zeros(len(adjList))
    start = max(tangentPlanes, key=lambda v: v.center[2])
    startIdx = tangentPlanes.index(start)
    visited[startIdx] = 1

    if np.dot(start.normal, np.array([0, 0, 1])) < 0:
        start.normal = -start.normal

    # makes tangentPlane2 have the same normal vec
    # orientation as tangentPlane1
    def align(tp1, tp2):
        if np.dot(tp1.normal, tp2.normal) < 0:
            tp2.normal = -tp2.normal

    def DFS_iterative(startIdx):
        stack = [startIdx]
        while stack:
            v = stack.pop()
            for n in adjList[v]:
                if not visited[n]:
                    visited[n] = 1
                    align(tangentPlanes[v], tangentPlanes[n])
                    stack.append(n)

    DFS_iterative(startIdx)


def getSDF(pc, tp, delta, ro):

    # pcTree = RTree(M=10, dim=3)
    #
    # for p in pc:
    #     pcTree.Insert(IndexRecord(bound=None, tuple_identifier=p))
    #
    tpTree = RTree(M=10, dim=3)

    for i in range(len(tp)):
        bound = Cube(np.repeat(tp[i].center, 2))
        tpTree.Insert(IndexRecord(bound=bound, tuple_identifier=i))

    def dist(point, tp):
        return np.dot(point - tp.center, tp.normal)

    def sdf(x):
        idx = tpTree.NearestNeighbor(
            IndexRecord(bound=Cube(np.repeat(x, 2)), tuple_identifier=0)
        )[0]
        tpNear = tp[idx]
        projNorm = dist(x, tpNear)
        # projX = x - projNorm * tpNear.normal
        #
        # zNN = pcTree.NearestNeighbor(IndexRecord(bound=None, tuple_identifier=projX))[
        #     0
        # ].tuple_identifier
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
        for i in range(len(self.pc)):
            self.pcTree.Insert(
                IndexRecord(bound=Cube(np.repeat(self.pc[i], 2)), tuple_identifier=i)
            )

    def computeTPs(self, k=15):
        self.tP = tangentPlanes(pc=self.pc, tree=self.pcTree, numNeighbors=k)
        self.centers = [p.center for p in self.tP]
        self.normals = [p.normal for p in self.tP]

        self.tpTree = RTree(M=10, dim=3)
        for i in range(len(self.centers)):
            bound = Cube(np.repeat(self.centers[i], 2))
            self.tpTree.Insert(IndexRecord(bound=bound, tuple_identifier=i))

        self.boundingBox = self.tpTree.root.covering

    def computeRiemanianGraph(self, k=15):
        self.rG = RiemanianGraph(self.centers, self.tpTree, k=k)

    def computeEMST(self):
        self.emst = EMST(self.centers, self.tpTree)

    def computeTraversalMST(self):
        self.rGEMSTjoin = GraphUtils.joinGraphs(self.rG.adjList, self.emst.adjList)
        self.mst = getMST(self.tP, self.rGEMSTjoin, OrientedTP.offset)

    def computeMesh(self):
        fixOrientations(self.tP, self.mst)
        self.normals = [p.normal for p in self.tP]
        self.sdf = getSDF(pc=self.pc, tp=self.tP, delta=0, ro=self.rG.longestEdge / 2)
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

    def visualizeEMST(self):
        return self.emst.visualizeEdges()

    def visualizeTraversalMST(self):
        return GraphUtils.visualizeEdges(self.centers, self.mst)

    def visualizeSurface(self):
        return self.mesh


def MarchingCubes(surface, length):

    def calcCubeIndex(x, y, z, sdfMesh):
        sdfs = sdfMesh[
            x + VertexPosition[:, 0], y + VertexPosition[:, 1], z + VertexPosition[:, 2]
        ]
        signs = sdfs >= 0
        return signs.dot(2 ** np.arange(8)), sdfs

    def interpolate(v1, v2, sdfV1, sdfV2):
        w1 = abs(sdfV1)
        w2 = abs(sdfV2)
        scale = w1 / (w1 + w2)
        v = scale * (v2 - v1) + v1
        return v

    def parseEdges(edges, pos, sdfs):
        rep = (len(edges) - 1) // 3 + 1
        for i in range(rep):
            idx = 3 * i
            if edges[idx] == -1:
                break

            v00, v01 = EdgeVertexIndices[edges[idx]]
            v10, v11 = EdgeVertexIndices[edges[idx + 1]]
            v20, v21 = EdgeVertexIndices[edges[idx + 2]]

            v0 = interpolate(
                pos + length * VertexPosition[v00],
                pos + length * VertexPosition[v01],
                sdfs[v00],
                sdfs[v01],
            )
            v1 = interpolate(
                pos + length * VertexPosition[v10],
                pos + length * VertexPosition[v11],
                sdfs[v10],
                sdfs[v11],
            )
            v2 = interpolate(
                pos + length * VertexPosition[v20],
                pos + length * VertexPosition[v21],
                sdfs[v20],
                sdfs[v21],
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
                    cubeIndex, sdfs = calcCubeIndex(i, j, k, sdfGrid3D)

                    edges = TriangleTable[cubeIndex]
                    parseEdges(edges, currCorner, sdfs)

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
