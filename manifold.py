import abc
import torch
import warnings
from typing import Optional

# Typing hints... ideally we would specify sizes here, but can't do that with the
# current type system
Point = torch.Tensor
Scalar = torch.Tensor
Vector = torch.Tensor


class Manifold(abc.ABC):

    # shape is the shape of 'points' on the manifold. For example, if the manifold
    # consists of 3x3 rotation matrices, then shape = (3, 3)
    shape: torch.Size
    # dim is the dimension of the manifold. For example, if the manifold consists
    # of 3x3 rotation matrices, then dim = 4. Dimensionality is essentially the
    # number of degrees of freedom of the manifold.
    dim: int
    # ambient is the dimension of the ambient space. For example, if the manifold
    # consists of 3x3 rotation matrices, then ambient = 9
    ambient: int

    @abc.abstractmethod
    def project(self, pt: Point) -> Point:
        """Project a point from the ambient space onto the manifold.

        :param pt: a point in the ambient space
        :return: a point on the manifold that is 'as close as possible' to pt
        """

    @abc.abstractmethod
    def contains(self, pt: Point, atol: float = 1e-6) -> bool:
        """Check whether the given point in the ambient space is on the manifold, plus
        or minus a bit of tolerance.
        """

    def distance(self, pt_x: Point, pt_y: Point) -> Scalar:
        """Compute distance-on-the-manifold from pt_x to pt_y

        :param pt_x: starting point in the space
        :param pt_y: ending point in the space
        :return: scalar length (or 'distance' or 'metric') from x to y
        """
        if not self.contains(pt_x):
            warnings.warn("pt_x is not on the manifold - trying to project")
            pt_x = self.project(pt_x)

        if not self.contains(pt_y):
            warnings.warn("pt_y is not on the manifold - trying to project")
            pt_y = self.project(pt_y)

        return self._distance_impl(pt_x, pt_y)

    @abc.abstractmethod
    def _distance_impl(self, pt_x: Point, pt_y: Point) -> Scalar:
        """Implementation of length(pt_x, pt_y) without checking for contains() first.
        """


    def geodesic(self, pt_x: Point, pt_y: Point, t: float) -> Point:
        """Compute the point along the geodesic from pt_x to pt_y at time t.

        :param pt_x: starting point in the space
        :param pt_y: ending point in the space
        :param t: fraction of distance from x to y
        :return: point along the geodesic from x to y
        """
        # Default implementation: exponential map of t * log map
        return self.exp_map(pt_x, self.log_map(pt_x, pt_y) * t)

    @abc.abstractmethod
    def to_tangent(self, pt_x: Point, vec_w: Vector) -> Vector:
        """Project a vector into the tangent space at pt_x.

        :param pt_x: point on the manifold
        :param vec_w: a vector in the ambient space whose base is at pt_x
        :return: projection of vec_w into the tangent space at pt_x
        """

    @abc.abstractmethod
    def inner_product(self, pt_x: Point, vec_w: Vector, vec_v: Vector):
        """Inner-product between two tangent vectors (defined at pt_x)

        :param pt_x: point defining the tangent space
        :param vec_w: first vector
        :param vec_v: second vector
        :return: inner product between w and v
        """

    def angle(self, pt_x: Point, vec_w: Vector, vec_v: Vector):
        """Angle between two tangent vectors

        :param pt_x: point defining the tangent space
        :param vec_w: first vector
        :param vec_v: second vector
        :return: angle between w and v
        """
        dot_wv = self.inner_product(pt_x, vec_w, vec_v)
        norm_w = self.norm(pt_x, vec_w)
        norm_v = self.norm(pt_x, vec_v)
        return torch.acos(torch.clip(dot_wv / (norm_w * norm_v), -1, 1))

    def squared_norm(self, pt_x: Point, vec_w: Vector):
        """Compute squared norm of a tangent vector at a point

        :param pt_x: point defining the tangent space
        :param vec_w: first vector
        :return: squared length of w according to the metric, AKA <w,w>
        """
        return self.inner_product(pt_x, vec_w, vec_w)

    def norm(self, pt_x: Point, vec_w: Vector):
        """Compute norm of a tangent vector at a point

        :param pt_x: point defining the tangent space
        :param vec_w: first vector
        :return: length of w according to the metric
        """
        return torch.sqrt(self.squared_norm(pt_x, vec_w))


    @abc.abstractmethod
    def exp_map(self, pt_x: Point, vec_w: Vector) -> Point:
        """Compute exponential map, which intuitively means finding the point pt_y
        that you get starting from pt_x and moving in the direction vec_w, which must
        be in the tangent space of pt_x.

        :param pt_x: base point
        :param vec_w: tangent vector
        :return: pt_y, the point you get starting from pt_x and moving in the
        direction vec_w
        """

    @abc.abstractmethod
    def log_map(self, pt_x: Point, pt_y: Point) -> Vector:
        """Compute logarithmic map, which can be thought of as the inverse of the
        exponential map.

        :param pt_x: base point. This defines where the tangent space is.
        :param pt_y: target point such that exp_map(pt_x, log_map(pt_x, pt_y)) = pt_y
        :return: vec_w, the vector in the tangent space at pt_x pointing in the
        direction (and magnitude) of pt_y
        """

    @abc.abstractmethod
    def levi_civita(self, pt_x: Point, pt_y: Point, vec_w: Vector) -> Vector:
        """Parallel-transport a tangent vector vec_w from pt_x to pt_y. The
        Levi-Civita connection is a nice way of defining "parallel lines" originating
        at two different places in a curved space. We say that vec_v at pt_y is
        parallel to vec_w at pt_x if, locally at b, levi_civita(pt_x, pt_y, vec_w) is
        colinear with vec_v.

        :param pt_x: base point where vec_w is a tangent vector
        :param pt_y: target point to transport to.
        :param vec_w: the tangent vector at pt_x to be transported to pt_y
        :return: vec_v, a vector in the tangent space of pt_y, corresponding to the
        parallel transport of vec_w
        """


__all__ = ["Point", "Scalar", "Vector", "Manifold"]
