import math
import random
import textwrap
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any

import numpy as np
from colorutils import Color

from .plot import plot_mesh, plot_points, plot_polygons
from .rtree_utils import IndexPointer, IndexRecord


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


class Node:

    def __init__(self, items, covering, level):

        self.items = items
        self.covering = covering
        self.level = level

    def add_entry(self, entry):
        """
        Add an entry to the node's list of items
        """
        return


class RTree(object):

    from .rtree_utils import Cube, Rect

    class BranchNode(Node):

        def __init__(self, items, covering, level=0):
            super().__init__(items, covering, level)

        def add_entry(self, entry):

            self.items.append(entry)

            if self.covering:
                self.covering = RTree.Bound.combine([self.covering, entry.bound])

        def update_bound(self, bound):
            self.covering = bound

        def plot(self, lp, bp, pp, colors):
            for i in self.items:
                i.pointer.plot(lp, bp, pp, colors)

            for f in self.covering.get_facets():
                for p in f:
                    bp.append(p)

        def __str__(self):
            string = ""
            for i in self.items:
                string += str(i) + "\n"
            return (
                "Branch "
                + f"{self.level} "
                + "(\n"
                + textwrap.indent(string, "    ")
                + ")"
            )

        def __repr__(self):
            string = ""
            for i in self.items:
                string += str(i) + "\n"
            return (
                "Branch "
                + f"{self.level} "
                + "(\n"
                + textwrap.indent(string, "    ")
                + ")"
            )

    class LeafNode(Node):

        def __init__(self, items, covering, level=0):

            super().__init__(items, covering, level)
            self.color = "#" + "".join(
                [random.choice("ABCDEF0123456789") for i in range(6)]
            )

        def plot(self, lp, bp, pp, colors):

            for f in self.covering.get_facets():
                for p in f:
                    lp.append(p)

            for v in self.items:
                pp.append(v.value)
            c = np.array([Color(web=self.color).rgb])
            cm = np.repeat(c, len(self.items), axis=0)
            for c in cm:
                colors.append(c)

        def add_entry(self, entry):

            self.items.append(entry)

            if self.covering:
                self.covering = RTree.Bound.combine([self.covering, entry.bound])

            else:
                self.covering = entry.bound

        def rm_entry(self, entry):

            for i in range(len(self.items)):
                if entry.__eq__(self.items[i]):

                    self.items.remove(entry)
                    self.update_bound(
                        RTree.Bound.combine([j.bound for j in self.items])
                    )
                    return True

            return False

        def update_bound(self, bound):
            self.covering = bound

        def __str__(self):
            string = ""
            for i in self.items:
                string += str(i) + "\n"
            return (
                "Leaf "
                + f"{self.level} ("
                + "\n"
                + textwrap.indent(string, "    ")
                + ")"
            )

        def __repr__(self):
            string = ""
            for i in self.items:
                string += str(i) + "\n"
            return (
                "Leaf "
                + f"{self.level} ("
                + "\n"
                + textwrap.indent(string, "    ")
                + ")"
            )

    ###########################################################################
    # Methods                                                                 #
    ###########################################################################

    def __init__(self, M, dim):

        self.max_num = M
        self.min_num = max(math.floor(M * 0.4), 1)

        self.height = 0
        self.p = max(min(math.floor(M * 0.3), 32), 1)
        self.dim = dim

        if self.dim == 2:
            RTree.Bound = RTree.Rect

        elif self.dim == 3:
            RTree.Bound = RTree.Cube

        self.root = RTree.LeafNode(items=[], covering=None, level=0)

    def plot(self):
        # should modify a list of plot objects lp, bp, pp which
        # stands for lineplot, branchplot, pointplot
        lp, bp, pp, colors = [], [], [], []
        self.root.plot(lp, bp, pp, colors)
        colors = np.array(colors)

        # We then plot them at the same time at the end
        branches = None

        if self.dim == 2:

            points = plot_points(points=pp, color=colors, dim=2)
            leaves = plot_polygons(vertices=lp, color="#009b00")
            if bp:
                branches = plot_polygons(vertices=bp, color="#ff0000")

        elif self.dim == 3:

            points = plot_points(points=pp, color=colors / 255, dim=3)
            leaves = plot_mesh(vertices=lp, color="#009b00")
            if bp:
                branches = plot_mesh(vertices=bp, color="#ff0000")

        return points, leaves, branches

    def __str__(self):
        return "Root:\n" + textwrap.indent(f"{self.root}", "    ")

    def __repr__(self):
        return "Root:\n" + textwrap.indent(f"{self.root}", "    ")

    def FindAddedArea(ptr, index_entry):

        exp_vol = RTree.Bound.expand_vol(ptr.bound, index_entry.bound)
        curr_vol = ptr.bound.vol
        diff = exp_vol - curr_vol

        return curr_vol, diff

    # returns current overlap area and the difference in overlap area when
    # adding new entry
    def FindAddedOverlap(ptr, ptrs, index_entry):

        curr_overlap = sum(ptr.bound.overlap(p.bound) for p in ptrs if p != ptr)
        new_bound = RTree.Bound.combine([ptr.bound, index_entry.bound])
        new_overlap = sum(new_bound.overlap(p.bound) for p in ptrs if p != ptr)
        diff = new_overlap - curr_overlap

        return curr_overlap, diff

    # choosing parent of entry to insert
    def ChooseSubTree(self, node, index_entry, curr_lvl):

        # pick the subdirectory that leads to least expansion
        min_exp, min_vol = math.inf, math.inf
        idx_ptr, ptr_pos = None, 0

        if curr_lvl == 0:

            # sort by least area needed to expand and take first p entries
            node.items = sorted(
                node.items, key=lambda x: RTree.FindAddedArea(x, index_entry)[1]
            )
            items = node.items[: self.p]

        else:
            items = node.items

        # find the expansion that results in least vol added
        for i in range(len(items)):
            curr_ptr = items[i]

            if curr_lvl == 0:
                curr_vol, diff = RTree.FindAddedOverlap(
                    curr_ptr, node.items, index_entry
                )

            else:
                curr_vol, diff = RTree.FindAddedArea(curr_ptr, index_entry)

            if diff < min_exp:

                min_exp = diff
                idx_ptr = curr_ptr
                ptr_pos = i
                min_vol = curr_vol

            # tiebreaker: choose smaller bounding box
            elif diff == min_exp:

                if curr_vol < min_vol:
                    idx_ptr = curr_ptr
                    ptr_pos = i
                    min_vol = curr_vol

        return idx_ptr, ptr_pos

    # takes in node, could be either leafnode or branchnode
    # and finds the axis to split along based on their items
    def ChooseSplitAxis(self, node):

        # splits items along start + idx index and calculates the
        # goodness value based on the two split lists
        def helper_func(items, start, idx):

            sb1 = [items[j].bound for j in range(start + idx)]
            sb2 = [
                items[j + start + idx].bound for j in range(len(items) - start - idx)
            ]
            mb1 = RTree.Bound.combine(sb1)
            mb2 = RTree.Bound.combine(sb2)
            margin_1, margin_2 = mb1.margin(), mb2.margin()

            return margin_1 + margin_2

        # Choose axis that will give us the lowest goodness value
        # calculated by the margin of the bounds
        g_value = math.inf

        # x-axis calculation
        # sort by lower value of bounding box
        x_l_sort = sorted(node.items, key=lambda x: x.bound.min_x)

        # x-axis calculation
        # sort by upper value f bounding box
        x_u_sort = sorted(node.items, key=lambda x: x.bound.max_x, reverse=True)

        # y-axis calculation
        # sort by upper value f bounding box
        y_l_sort = sorted(node.items, key=lambda x: x.bound.min_y)

        # y-axis calculation
        # sort by upper value f bounding box
        y_u_sort = sorted(node.items, key=lambda x: x.bound.max_y, reverse=True)

        if self.dim == 3:

            # z-axis calculation
            # sort by upper value f bounding box
            z_l_sort = sorted(node.items, key=lambda x: x.bound.min_z)

            # z-axis calculation
            # sort by upper value f bounding box
            z_u_sort = sorted(node.items, key=lambda x: x.bound.max_z, reverse=True)

        for i in range(self.max_num - 2 * self.min_num + 1):

            curr_g = helper_func(x_l_sort, self.min_num, i)
            if curr_g < g_value:
                g_value = curr_g
                node.items = x_l_sort

            curr_g = helper_func(x_u_sort, self.min_num, i)
            if curr_g < g_value:
                g_value = curr_g
                node.items = x_u_sort

            curr_g = helper_func(y_l_sort, self.min_num, i)
            if curr_g < g_value:
                g_value = curr_g
                node.items = y_l_sort

            curr_g = helper_func(y_u_sort, self.min_num, i)
            if curr_g < g_value:
                g_value = curr_g
                node.items = y_u_sort

            if self.dim == 3:

                curr_g = helper_func(z_l_sort, self.min_num, i)
                if curr_g < g_value:
                    g_value = curr_g
                    node.items = z_l_sort

                curr_g = helper_func(z_u_sort, self.min_num, i)
                if curr_g < g_value:
                    g_value = curr_g
                    node.items = z_u_sort

    def ChooseSplitIndex(self, items):

        split_idx, b1, b2 = None, None, None
        min_overlap = math.inf
        min_vol = math.inf

        for i in range(self.max_num - 2 * self.min_num + 1):

            # find index to split along by least overlap
            s1 = [items[j].bound for j in range(self.min_num + i)]
            s2 = [
                items[j + self.min_num + i].bound
                for j in range(len(items) - self.min_num - i)
            ]

            tmp_b1 = RTree.Bound.combine(s1)
            tmp_b2 = RTree.Bound.combine(s2)
            curr_overlap = tmp_b1.overlap(tmp_b2)

            if curr_overlap <= min_overlap:
                curr_vol = tmp_b1.vol + tmp_b2.vol

                # tiebreaker: choose smaller bounding box
                if curr_overlap == min_overlap:

                    if curr_vol < min_vol:
                        split_idx = self.min_num + i
                        min_vol = curr_vol
                        b1, b2 = tmp_b1, tmp_b2

                else:

                    min_vol = curr_vol
                    min_overlap = curr_overlap
                    split_idx = self.min_num + i
                    b1, b2 = tmp_b1, tmp_b2

        return items[:split_idx], items[split_idx:], b1, b2

    def Split(self, node, lvl):

        self.ChooseSplitAxis(node)
        l1, l2, b1, b2 = self.ChooseSplitIndex(node.items)

        if lvl == 0:

            n1 = RTree.LeafNode(items=l1, covering=b1, level=lvl)
            n2 = RTree.LeafNode(items=l2, covering=b2, level=lvl)
            return n1, n2

        else:

            n1 = RTree.BranchNode(items=l1, covering=b1, level=lvl)
            n2 = RTree.BranchNode(items=l2, covering=b2, level=lvl)
            return n1, n2

    def OverflowTreatment(self, node, level, overflow):

        if level != self.height and level not in overflow:

            overflow.add(level)
            to_insert = self.Reinsert(node)
            return None, None, to_insert, level

        else:

            n1, n2 = self.Split(node, level)
            return n1, n2, None, None

    def Reinsert(self, node):

        sort_dist = sorted(
            node.items,
            key=lambda x: np.linalg.norm(node.covering.center - x.bound.center),
            reverse=True,
        )
        node.items = sort_dist[self.p :]
        node.update_bound(RTree.Bound.combine([n.bound for n in node.items]))

        return sort_dist[: self.p][::-1]

    def ChooseLeaf(self, node, index_entry, curr_lvl, overflowed, ins_lvl=0):

        n1, n2, q, r_lvl = None, None, [], None

        if curr_lvl == ins_lvl:
            node.add_entry(index_entry)

            if len(node.items) > self.max_num:
                if curr_lvl == 0:

                    # if node is too big, split leaf node
                    n1, n2, r, r_lvl = self.OverflowTreatment(
                        node,
                        curr_lvl,
                        overflowed,
                    )

                    if r:
                        for elem in r:
                            q.append((elem, r_lvl))

        else:

            # choosing parent of entry to insert
            idx_ptr, ptr_pos = self.ChooseSubTree(
                node,
                index_entry,
                curr_lvl,
            )

            # index.pointer is the pointer to one of node's children nodes
            n1, n2, q, r_lvl = self.ChooseLeaf(
                node=idx_ptr.pointer,
                index_entry=index_entry,
                curr_lvl=curr_lvl - 1,
                ins_lvl=ins_lvl,
                overflowed=overflowed,
            )

            # update bound
            idx_ptr.update(idx_ptr.pointer.covering)
            node.update_bound(RTree.Bound.combine([n.bound for n in node.items]))

            if n2:

                # Should be creating new indexpointers for each split created
                # These indexpointers will lie in our current node.
                node.items[ptr_pos] = IndexPointer(n2.covering, n2)
                other_ptr = IndexPointer(n1.covering, n1)
                node.add_entry(other_ptr)
                n2 = None

            if len(node.items) > self.max_num:

                # If the branch node has too many items split
                n1, n2, r, r_lvl = self.OverflowTreatment(
                    node,
                    curr_lvl,
                    overflowed,
                )

                if r:
                    for elem in r:
                        q.append((elem, r_lvl))

        # returns either split leaf nodes, or branch nodes, depending on
        # which one is at the highest level of tree
        return n1, n2, q, r_lvl

    # inserts entry with propagation of changes upwards until root node
    # if root node is too big, we split
    def Insert(self, entry, ins_lvl=0, overflowed=set()):

        n1, n2, q, r_lvl = self.ChooseLeaf(
            node=self.root,
            index_entry=entry,
            curr_lvl=self.height,
            ins_lvl=ins_lvl,
            overflowed=overflowed,
        )

        if n2:

            p1 = IndexPointer(bound=n1.covering, pointer=n1)
            p2 = IndexPointer(bound=n2.covering, pointer=n2)
            new_bound = RTree.Bound.combine([p1.bound, p2.bound])

            self.height += 1
            self.root = RTree.BranchNode(
                items=[],
                covering=new_bound,
                level=self.height,
            )
            self.root.add_entry(p1)
            self.root.add_entry(p2)

        if q:
            for pairs in q:
                self.Insert(pairs[0], ins_lvl=pairs[1], overflowed=overflowed)

    # removes entry from the tree
    def Delete(self, entry):

        def FindLeaf(node, index_entry, curr_lvl, ins_lvls, to_insert):

            if curr_lvl == 0:

                # returns whether entry is removed after attempting
                # to remove it
                return node.rm_entry(index_entry)

            else:

                # Set of index records to be readded in case of
                # underfull node after deletion
                rm_item, ins_lvl = False, 0

                # Findleaf on all branches that might have index_entry
                for i in range(len(node.items) - 1, -1, -1):
                    curr_item = node.items[i]

                    if curr_item.bound.contains(index_entry.bound):
                        child_node = curr_item.pointer

                        # If we found an entry and deleted it, do whats after
                        if FindLeaf(
                            child_node,
                            index_entry,
                            curr_lvl=curr_lvl - 1,
                            ins_lvls=ins_lvls,
                            to_insert=to_insert,
                        ):
                            rm_item = True

                            # delete underfull nodes
                            if len(child_node.items) < self.min_num:

                                q = child_node.items
                                ins_lvl = child_node.level
                                del node.items[i]

                                # Remember to reinsert later
                                for elem in q:
                                    ins_lvls.append(ins_lvl)
                                    to_insert.append(elem)

                            else:

                                # fix indexpointer covering
                                curr_item.bound = child_node.covering

                    # points to branch, update bound if childpointer was changed
                    if rm_item:
                        node.update_bound(
                            RTree.Bound.combine([n.bound for n in node.items])
                        )

                        return True

                return False

        # Store insert levels and node to insert in different lists:
        insert_lvls, to_insert = [], []

        # call recursive function
        FindLeaf(
            self.root,
            index_entry=entry,
            curr_lvl=self.height,
            ins_lvls=insert_lvls,
            to_insert=to_insert,
        )

        # reinsert here
        for lvl, node in zip(insert_lvls, to_insert):
            self.Insert(entry=node, ins_lvl=lvl)

        # Fix the root node if there are too few entries in root.items
        if len(self.root.items) < 2 and self.height != 0:

            q = []
            items = self.root.items

            for i in range(len(items) - 1, 0, -1):

                q += items[i].pointer.items
                del self.root.items[i]

            self.root = items[0].pointer
            self.height -= 1

            # reinsert here
            if self.height == 0:
                ins_lvl = 0

            else:
                ins_lvl = self.height

            for elem in q:
                self.Insert(entry=elem, ins_lvl=ins_lvl)

    # given a scope (search rectangle/Bound) returns list of index records
    # contained in that scope
    def Search(self, scope):

        found = []

        def helper_func(node, found):

            if type(node) is RTree.LeafNode:
                for record in node.items:
                    if scope.contains(record.bound):
                        found.append(record)

            else:
                for b in node.items:
                    if scope.overlap(b.bound):
                        helper_func(b.pointer, found)

        helper_func(self.root, found)
        return found

    # given a scope (search rectangle/Bound) returns list of index records
    # contained in that scope
    def SearchOverlap(self, scope):

        found = []

        def helper_func(node, found):

            if type(node) is RTree.LeafNode:
                for record in node.items:
                    if scope.overlap(record.bound) > 0:
                        found.append(record)

            else:
                for b in node.items:
                    if scope.overlap(b.bound):
                        helper_func(b.pointer, found)

        helper_func(self.root, found)
        return found

    def NearestNeighbor(self, entry, k=1):

        pq = PriorityQueue()
        pq.put(PrioritizedItem(0, self.root))
        neighbors = []

        while not pq.empty():
            elem = pq.get().item

            if isinstance(elem, IndexRecord):
                neighbors.append(elem.tuple_identifier)

                if len(neighbors) == k:
                    return neighbors

            elif type(elem) is RTree.LeafNode:
                for r in elem.items:

                    dist_r = np.linalg.norm(r.bound.bound[::2] - entry.bound.bound[::2])
                    e = PrioritizedItem(dist_r, r)
                    pq.put(e)

            else:
                for b in elem.items:

                    child_node = b.pointer
                    dist = RTree.Bound.get_dist(
                        child_node.covering, entry.bound.bound[::2]
                    )
                    e = PrioritizedItem(dist, child_node)
                    pq.put(e)

        return neighbors
