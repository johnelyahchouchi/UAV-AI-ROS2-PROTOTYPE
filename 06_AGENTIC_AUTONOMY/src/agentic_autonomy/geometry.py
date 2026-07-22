from __future__ import annotations

from math import hypot

from .domain import Point


def distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def polygon_centroid(vertices: tuple[Point, ...]) -> Point:
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        cross = a.x * b.y - b.x * a.y
        area2 += cross
        cx += (a.x + b.x) * cross
        cy += (a.y + b.y) * cross
    if abs(area2) < 1e-12:
        raise ValueError("polygon area must be non-zero")
    return Point(cx / (3.0 * area2), cy / (3.0 * area2))


def point_in_polygon(point: Point, vertices: tuple[Point, ...]) -> bool:
    inside = False
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        cross = (point.x - a.x) * (b.y - a.y) - (point.y - a.y) * (b.x - a.x)
        if abs(cross) < 1e-9 and min(a.x, b.x) - 1e-9 <= point.x <= max(a.x, b.x) + 1e-9 and min(a.y, b.y) - 1e-9 <= point.y <= max(a.y, b.y) + 1e-9:
            return True
        if (a.y > point.y) != (b.y > point.y):
            x_intersection = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x
            if point.x < x_intersection:
                inside = not inside
    return inside

