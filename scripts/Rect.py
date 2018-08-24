import numpy as np


class Rect:
    def __init__(self, p, height, width):
        self.x = np.array(p)
        self.h = height
        self.w = width

    def get_parameters(self):
        return self.x, self.h, self.w

    def is_crossing(self, rect):
        rect_points = [
            rect.x,
            rect.x + np.array([rect.h, 0]),
            rect.x + np.array([0, rect.w]),
            rect.x + np.array([rect.h, rect.w])
        ]

        self_points = [
            self.x,
            self.x + np.array([self.h, 0]),
            self.x + np.array([0, self.w]),
            self.x + np.array([self.h, self.w])
        ]

        ok = 0
        for x in rect_points:
            ok |= self.point_in(x)

        for x in self_points:
            ok |= rect.point_in(x)

        return ok

    def get_convex_rect(self, rect):
        rect_points = [
            rect.x,
            rect.x + np.array([rect.h, 0]),
            rect.x + np.array([0, rect.w]),
            rect.x + np.array([rect.h, rect.w])
        ]

        self_points = [
            self.x,
            self.x + np.array([self.h, 0]),
            self.x + np.array([0, self.w]),
            self.x + np.array([self.h, self.w])
        ]

        all_points = np.array(self_points + rect_points)

        maxx = np.max(all_points[:, 0])
        minx = np.min(all_points[:, 0])
        maxy = np.max(all_points[:, 1])
        miny = np.min(all_points[:, 1])

        return Rect((minx, miny), maxx - minx, maxy - miny)

    def point_in(self, p):
        return self.x[0] <= p[0] <= self.x[0] + self.h and self.x[1] <= p[1] <= self.x[1] + self.w
