import torch
from manifold import Point, Manifold, Vector, Scalar


class EuclideanSpace(Manifold):
    def project(self, pt: Point) -> Point:
        return pt

    def contains(self, pt: Point, atol: float = 1e-6) -> bool:
        return True

    def _distance_impl(self, pt_x: Point, pt_y: Point) -> Scalar:
        return torch.linalg.norm(pt_x - pt_y)

    def to_tangent(self, pt_x: Point, vec_w: Vector) -> Vector:
        return vec_w

    def inner_product(self, pt_x: Point, vec_w: Vector, vec_v: Vector):
        return torch.dot(vec_w, vec_v)

    def exp_map(self, pt_x: Point, vec_w: Vector) -> Point:
        return pt_x + vec_w

    def log_map(self, pt_x: Point, pt_y: Point) -> Vector:
        return pt_y - pt_x

    def levi_civita(self, pt_x: Point, pt_y: Point, vec_w: Vector) -> Vector:
        return vec_w
