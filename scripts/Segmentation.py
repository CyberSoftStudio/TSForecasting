from scripts.extremumlib import *
import numpy as np
from math import atan2


def point_compare(a, b):
    if a[1] * b[0] == b[1] * a[0]:
        return a[1] ** 2 + a[0] ** 2 - (b[1] ** 2 + b[0] ** 2)
    return a[1] * b[0] - b[1] * a[0]


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def cross_product(a, b):
    return a[0] * b[1] - a[1] * b[0]


class Segment:
    def __init__(self, points=[]):
        self.points = points
        self.convex = []
        self.convex_rect = []

    def recalc_convex_rect(self):
        minx = self.points[0][0]
        miny = self.points[0][1]
        maxx = self.points[0][0]
        maxy = self.points[0][1]

        for i in range(1, len(self.points)):
            minx = min(minx, self.points[i][0])
            miny = min(miny, self.points[i][1])
            maxx = max(maxx, self.points[i][0])
            maxy = max(maxy, self.points[i][1])

        self.convex_rect = []
        self.convex_rect.append((minx, miny))
        self.convex_rect.append((maxx, miny))
        self.convex_rect.append((maxx, maxy))
        self.convex_rect.append((minx, maxy))
        self.convex_rect.append((minx, miny))   # closure

        self.convex_rect = np.array(self.convex_rect)

    def recalc_convex_hull(self):
        points = list(np.array(self.points).copy())
        lp = min(((x[0], x[1]) for x in points))
        print("minimal point ", lp)
        # lp = (lp - 1, 0)
        points = sorted(points, key=cmp_to_key(point_compare))
        convex = [lp]
        if len(points) < 2:
            self.points = np.array(self.points)
            self.convex = np.array(convex)
            return self

        convex = [points[0], points[1]]

        for i in range(2, len(points)):
            a = np.array(convex[-2])
            b = np.array(convex[-1])
            c = np.array(points[i])

            if cross_product(b - a, c - b) < 0:
                while len(convex) > 1 and cross_product(b - a, c - b) <= 0:
                    # print("i am in while")
                    convex.pop()
                    if len(convex) == 1:
                        break
                    a = np.array(convex[-2])
                    b = np.array(convex[-1])
                # print("while finished")
            convex.append(c)
        # convex.append(convex[0])
        self.convex = np.array(convex).copy()

    def normalize(self):

        # self.points = np.array(self.points)
        # return self
        # self.recalc_convex_hull()

        self.points = np.array(self.points)
        self.recalc_convex_rect()
        return self


class Segmentation:
    def __init__(self, plane):
        self.plane_orig = plane
        self.plane = None
        self.used = np.zeros(plane.shape)
        self.n = plane.shape[0]
        self.m = plane.shape[1]
        self.segmentation = []
        self.separators = []

    def prepare(self, alpha = 0.25):
        return bound_filter(linear(self.plane_orig), alpha=alpha)

    def extract(self, alpha=0.25):
        self.plane = self.prepare(alpha=alpha)
        n, m = self.plane.shape
        print(n, m)
        for i in range(n):
            for j in range(m):
                if self._ok((i, j)):
                    s = Segment([])
                    self.dfs((i, j), s)
                    print(s.points)
                    self.segmentation.append(s.normalize())

    def _ok(self, coord):
        return (0 <= coord[0] < self.n) and (0 <= coord[1] < self.m) and (self.used[coord[0], coord[1]] != True) and abs(self.plane[coord[0], coord[1]]) > 1e-4

    def dfs(self, p, segment):
        # print("dfs in point ", p)
        if not self._ok(p):
            return

        self.used[p[0], p[1]] = True

        segment.points.append(p)

        steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for step in steps:
            np = (p[0] + step[0], p[1] + step[1])
            if self._ok(np):
                # print("new point ", np)
                self.dfs(np, segment)

    def show(self):
        result = self.plane_orig.copy()
        paths = [x.convex_rect for x in self.segmentation]
        # print("paths length", len(paths))
        return result, paths

    def recalc_separators(self):
        self.segmentation = sorted(self.segmentation, key = lambda a: a.convex_rect[0][1])
        print([s.convex_rect for s in self.segmentation])
        self.separators = []
        print("number of segments ", len(self.segmentation))
        for i in range(1, len(self.segmentation)):
            print(self.segmentation[i - 1].convex_rect[1][1], self.segmentation[i].convex_rect[0][1])
            self.separators.append((self.segmentation[i - 1].convex_rect[2][1] + self.segmentation[i].convex_rect[0][1])/2)

        self.separators = np.array(self.separators)

    def get_intervals(self):
        intervals = []
        for s in self.segmentation:
            s.recalc_convex_rect()
            x, y = (s.convex_rect[0] + s.convex_rect[2]) // 2
            intervals.append((s.convex_rect[0][1], s.convex_rect[2][1], np.sign(self.plane[x, y])))
        return intervals
