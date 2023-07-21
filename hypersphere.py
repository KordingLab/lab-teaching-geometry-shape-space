import torch
import numpy as np
from manifold import Point, Vector, Scalar, Manifold


def slerp(pt_x: Point, pt_y: Point, frac: float) -> Point:
    """Spherical Linear intERPolation between two points -- see [1]. The interpolated
    point will always have unit norm.

    [1] https://en.m.wikipedia.org/wiki/Slerp

    :param pt_x: starting point. Will be normalized to unit length.
    :param pt_y: ending point. Will be normalized to unit length.
    :param frac: fraction of arc length from pt_x to pt_y
    :return: unit vector on the great-circle connecting pt_x to pt_y that is 'frac'
    of the distance from pt_x to pt_y
    """
    assert 0.0 <= frac <= 1.0, "frac must be between 0 and 1"

    def _norm(vec):
        return vec / torch.sqrt(torch.sum(vec * vec))

    # Normalize a and b to unit vectors
    a, b = _norm(pt_x), _norm(pt_y)

    # Check cases where we can break early (and doing so avoids things like divide-by-zero later!)
    if frac == 0.0:
        return a
    elif frac == 1.0:
        return b

    # Use dot product between (normed) a and b to test for colinearity
    dot_ab = torch.sum(a * b)

    # Check some more break-early cases based on dot product result.
    eps = 1e-6
    if dot_ab > 1.0 - eps:
        # dot(a,b) is effectively 1, so A and B are effectively the same vector. Do
        # Euclidean interpolation.
        return _norm(a*(1-frac) + b*frac)
    elif dot_ab < -1 + eps:
        # dot(a,b) is effectively -1, so A and B are effectively at opposite poles.
        # There are infinitely many geodesics.
        raise ValueError("A and B are andipodal - cannot SLERP")

    # Get 'omega' - the angle between a and b, clipping for numerical stability
    omega = torch.acos(torch.clip(dot_ab, -1.0, 1.0))
    # Do interpolation using the SLERP formula
    a_frac = a * torch.sin((1 - frac) * omega) / torch.sin(omega)
    b_frac = b * torch.sin(frac * omega) / torch.sin(omega)
    return (a_frac + b_frac).reshape(a.shape)


class HyperSphere(Manifold):
    """Class for handling geometric operations on an n-dimensional hypersphere
    """

    def __init__(self, dim):
        # a dim-dimensional sphere has points that live in dim+1-dimensional space
        self.shape = torch.Size((dim+1,))
        self.dim = dim
        self.ambient = dim+1

    def geodesic(self, pt_x: Point, pt_y: Point, t: float) -> Point:
        return slerp(pt_x, pt_y, t)

    def project(self, pt: Point) -> Point:
        return pt / torch.sqrt(torch.sum(pt * pt))

    def contains(self, pt: Point, atol: float = 1e-6) -> bool:
        radius = torch.sqrt(torch.sum(pt * pt, dim=-1))
        return torch.abs(radius - 1.0) <= atol

    def _distance_impl(self, pt_x: Point, pt_y: Point) -> Scalar:
        dot_ab = torch.dot(pt_x, pt_y)
        len_a, len_b = torch.sqrt(torch.dot(pt_x, pt_x)), torch.sqrt(torch.dot(pt_y, pt_y))
        cosine = dot_ab / len_a / len_b
        return torch.arccos(torch.clip(cosine, -1.0, +1.0))

    def to_tangent(self, pt_x: Point, vec_w: Vector) -> Vector:
        dot_a_w = torch.sum(pt_x * vec_w)
        return vec_w - dot_a_w * pt_x

    def inner_product(self, pt_x: Point, vec_w: Vector, vec_v: Vector):
        # Just the usual inner product in the ambient space
        return torch.sum(vec_w * vec_v)

    def exp_map(self, pt_x: Point, vec_w: Vector) -> Point:
        # See https://math.stackexchange.com/a/1930880
        vec_w = self.to_tangent(pt_x, vec_w)
        norm = torch.sqrt(torch.sum(vec_w * vec_w))
        c1 = torch.cos(norm)
        c2 = torch.sinc(norm / np.pi)
        return c1 * pt_x + c2 * vec_w

    def log_map(self, pt_x: Point, pt_y: Point) -> Vector:
        unscaled_w = self.to_tangent(pt_x, pt_y)
        norm_w = unscaled_w / torch.clip(torch.sqrt(torch.sum(unscaled_w * unscaled_w)), 1e-7)
        return norm_w * self.distance(pt_x, pt_y)

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # Idea: decompose the tangent vector w into (i) a part that is orthogonal to the transport direction, and (ii)
        # a part along the transport direction. The orthogonal part will be unchanged through the map, and the parallel
        # part will be rotated in the plane spanned by pt_a and the unit v. (thanks to the geomstats package for
        # reference implementation)
        vec_v = self.log_map(pt_a, pt_b)
        angle = self.distance(pt_a, pt_b)
        unit_v = vec_v / torch.clip(angle, 1e-7)  # the length of tangent vector v *is* the length from a to b
        w_along_v = torch.sum(unit_v * vec_w)
        orth_part = vec_w - w_along_v * unit_v
        return orth_part + torch.cos(angle) * w_along_v * unit_v - torch.sin(angle) * w_along_v * pt_a
