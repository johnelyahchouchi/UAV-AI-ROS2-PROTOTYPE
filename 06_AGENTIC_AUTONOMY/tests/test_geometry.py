from agentic_autonomy.domain import Point
from agentic_autonomy.geometry import distance, point_in_polygon, polygon_centroid


def test_distance_and_centroid():
    assert distance(Point(0, 0), Point(3, 4)) == 5
    square = (Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2))
    assert polygon_centroid(square) == Point(1, 1)


def test_polygon_includes_boundary_and_excludes_outside():
    square = (Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2))
    assert point_in_polygon(Point(1, 1), square)
    assert point_in_polygon(Point(0, 1), square)
    assert not point_in_polygon(Point(3, 1), square)

