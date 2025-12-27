"""Box class for representing a d-dimensional box in the objective space."""

import numpy as np


class Box:
    """A d-dimensional box."""

    def __init__(self, point1, point2):
        """Initialize the box with two points."""
        self.dimensions = len(point1)
        self.bounds = np.array([point1, point2])
        self.nadir = np.min(self.bounds, axis=0)
        self.ideal = np.max(self.bounds, axis=0)
        self.midpoint = (self.nadir + self.ideal) / 2
        self.volume = self.compute_volume()
        self.max_dist = np.max(self.ideal - self.nadir)

    def compute_volume(self):
        """Compute the volume of the box.

        Returns:
            float: The volume of the box.
        """
        return abs(np.prod(self.ideal - self.nadir))

    def get_intersecting_box(self, box):
        """If the box intersect, construct the intersecting box.

        Args:
            box (Box): The box.

        Returns:
            Box: The intersecting box.
        """
        if not self.is_intersecting(box):
            return None
        return Box(np.max([self.nadir, box.nadir], axis=0), np.min([self.ideal, box.ideal], axis=0))

    def is_intersecting(self, box):
        """If the box intersect, construct the intersecting box.

        Note:
            To check if two boxes intersect, we can compare the ranges of each dimension for the two boxes. If there is
            any dimension where the ranges of the two boxes do not overlap, then the boxes do not intersect. If the
            ranges of all dimensions overlap, then the boxes intersect.

            For example, suppose we have two boxes in three dimensions:
            Box A: [x1, x2] x [y1, y2] x [z1, z2]
            Box B: [u1, u2] x [v1, v2] x [w1, w2]

            The two boxes intersect if and only if:
            x1 < u2 and u1 < x2 (overlap in the x dimension)
            y1 < v2 and v1 < y2 (overlap in the y dimension)
            z1 < w2 and w1 < z2 (overlap in the z dimension)

        Args:
            box (Box): The box.

        Returns:
            Box: The intersecting box.
        """
        return np.all(np.logical_and(self.nadir < box.ideal, box.nadir < self.ideal))

    def is_intersecting_with_boundary(self, box):
        """Check if the box intersect with the boundary included.

        Args:
            box (Box): The box.

        Returns:
            bool: True if the box intersect with the boundary.
        """
        return np.any(np.logical_and(self.nadir <= box.ideal, box.nadir <= self.ideal))

    def projection_is_intersecting(self, box, dim):
        """Check if the projection of the box where a dimension is removed is intersecting.

        Args:
            box (Box): The box.
            dim (int): The dimension to remove.

        Returns:
            bool: True if the projection is intersecting.
        """
        projected_nadir = np.delete(self.nadir, dim)
        projected_ideal = np.delete(self.ideal, dim)
        projected_box_nadir = np.delete(box.nadir, dim)
        projected_box_ideal = np.delete(box.ideal, dim)
        return np.all(np.logical_and(projected_nadir < projected_box_ideal, projected_box_nadir < projected_ideal))

    def contains(self, point):
        """Check if the point is in the box including the boundary.

        Args:
            point (Tuple[float]): The point.

        Returns:
            bool: True if the point is in the box.
        """
        return np.all(np.logical_and(self.nadir <= point, point <= self.ideal))

    def contains_inner(self, point):
        """Check if the point is in the box not including the boundary.

        Args:
            point (Tuple[float]): The point.

        Returns:
            bool: True if the point is in the box.
        """
        return np.all(np.logical_and(self.nadir < point, point < self.ideal))

    def vertices(self):
        """Compute the vertices of the box.

        Returns:
            List[Tuple[float]]: The vertices of the box.
        """
        vertices = []
        for i in range(2**self.dimensions):
            vertex = []
            for j in range(self.dimensions):
                if i & (1 << j):
                    vertex.append(self.ideal[j])
                else:
                    vertex.append(self.nadir[j])
            vertices.append(tuple(vertex))
        return vertices

    def __repr__(self):
        """String representation of the box."""
        return f"Box({self.nadir}, {self.ideal})"
