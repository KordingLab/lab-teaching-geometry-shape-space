import torch
import numpy as np
from torch.linalg import norm, eigh
from scipy.linalg import solve_sylvester
from manifold import Manifold, Point, Scalar, Vector
from hypersphere import slerp


class PreShapeSpace(Manifold):
    """The Pre Shape Metric is like the Shape Metric but without removing rotations (
    no alignment step).
    """

    def __init__(self, num_keypoints, keypoints_dim):
        self.m, self.p = num_keypoints, keypoints_dim
        self.shape = torch.Size((self.m, self.p))
        self.dim = self.p*(self.m-1) - (self.p*(self.p-1))//2
        self.ambient = self.m * self.p

    def project(self, pt: Point) -> Point:
        # Assume that pt.shape == (m, p). The first operation is to center it:
        pt = pt - torch.mean(pt, dim=0)
        # Remove scale by normalize the point
        pt = pt / norm(pt, ord="fro")
        return pt

    def contains(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test shape
        if pt.shape != self.shape:
            return False
        # Test centered
        if not torch.allclose(torch.mean(pt, dim=0), pt.new_zeros(pt.shape[1:]), atol=atol):
            return False
        # Test unit norm
        if not torch.isclose(norm(pt, ord="fro"), pt.new_ones((1,))):
            return False
        # All tests passed - return True
        return True

    def _distance_impl(self, pt_x: Point, pt_y: Point) -> Scalar:
        cos_ab = torch.sum(pt_x * pt_y) / torch.sqrt(torch.sum(pt_x * pt_x) * torch.sum(pt_y * pt_y))
        return torch.arccos(torch.clip(cos_ab, -1.0, +1.0))

    def geodesic(self, pt_x: Point, pt_y: Point, t: float) -> Point:
        return slerp(pt_x, pt_y, t)

    def to_tangent(self, pt_x: Point, vec_w: Vector):
        """Project to tangent in the pre-shape space. Pre-shapes are equivalent
        translation and scale but not rotation.

        :param pt_x: base point for the tangent vector
        :param vec_w: ambient space vector
        :return: tangent vector with mean-shifts removed, as well as scaling removed
        """
        # Points must be 'centered', so subtract off component of vec that would
        # affect the mean
        vec_w = vec_w - torch.mean(vec_w, dim=0)
        # Subtract off component that would uniformly scale all points (component of
        # the tangent in the direction of pt_x)
        vec_w = vec_w - pt_x * torch.sum(vec_w * pt_x) / torch.sum(pt_x * pt_x)
        return vec_w

    def inner_product(self, pt_x: Point, vec_w: Vector, vec_v: Vector):
        return torch.sum(vec_w * vec_v)

    def exp_map(self, pt_x: Point, vec_w: Vector) -> Point:
        # Identical to Hypersphere.exp_map
        # See https://math.stackexchange.com/a/1930880
        norm = self.norm(pt_x, vec_w)
        c1 = torch.cos(norm)
        c2 = torch.sinc(norm / np.pi)
        return c1 * pt_x + c2 * vec_w

    def log_map(self, pt_x: Point, pt_y: Point) -> Vector:
        # Identical to Hypersphere.log_map
        unscaled_w = self.to_tangent(pt_x, pt_y)
        norm_w = unscaled_w / torch.clip(self.norm(pt_x, unscaled_w), 1e-7)
        return norm_w * self.distance(pt_x, pt_y)

    def levi_civita(self, pt_x: Point, pt_y: Point, vec_w: Vector) -> Vector:
        # Identical to Hypersphere.levi_civita
        vec_v = self.log_map(pt_x, pt_y)
        angle = self.distance(pt_x, pt_y)
        unit_v = vec_v / torch.clip(angle, 1e-7)  # the length of tangent vector v *is* the length from x to y
        w_along_v = torch.sum(unit_v * vec_w)
        orth_part = vec_w - w_along_v * unit_v
        return orth_part + w_along_v * (torch.cos(angle) * unit_v - torch.sin(angle) * pt_x)


class ShapeSpace(PreShapeSpace):
    """Practical differences between PreShape and Shape:
    - Shape space decomposes the PreShape tangent space into vertical (within equivalence class) and horizontal (across
        equivalence class) parts.
    - Shape.to_tangent is not overridden, so Shape.to_tangent(pt, vec) will in general contain both horz and vert parts
    - Shape.exp_map and Shape.levi_civita both respect the vertical part
    - Shape.log_map returns *only* the horizontal part
    - Shape.inner_product only takes the horizontal part
    This means that exp_map and log_map are not exact inverses up to _equality_. However, they are inverses up to
    _equivalence_.
    """

    def _distance_impl(self, pt_x: Point, pt_y: Point) -> Scalar:
        # Distance in shape space = distance in pre shape space after aligning points
        # to each other
        pt_x, pt_y = _orthogonal_procrustes(pt_x, pt_y)
        return super(ShapeSpace, self)._distance_impl(pt_x, pt_y)

    def geodesic(self, pt_x: Point, pt_y: Point, t: float) -> Point:
        # Choice of anchor here is largely arbitrary, but for local consistency with
        # log_map we set it to 'x'
        pt_x, pt_y = _orthogonal_procrustes(pt_x, pt_y, anchor="x")
        return super(ShapeSpace, self).geodesic(pt_x, pt_y, t)

    def _horizontal_tangent(self, pt_x: Point, vec_w: Vector, *, vert_part: Vector = None) -> Vector:
        """The 'horizontal' part of the tangent space is the part that is actually
        movement in the quotient space, i.e. across equivalence classes. For example,
        east/west movement where equivalence = lines of longitude.
        """
        # Start by ensuring vec_w is a tangent vector in the pre-shape space
        vec_w = super(ShapeSpace, self).to_tangent(pt_x, vec_w)
        if vert_part is None:
            # Calculate vertical part
            vert_part = self._vertical_tangent(pt_x, vec_w)
        # The horizontal part is what is left after projecting away the vertical part
        square_vert_norm = torch.clip(torch.sum(vert_part * vert_part), 1e-7)
        horz_part = vec_w - vert_part * torch.sum(vec_w * vert_part) / square_vert_norm
        return horz_part

    def _solve_skew_symmetric_vertical_tangent(self, pt_x: Point, vec_w: Vector):
        """Find A such that x@A is the vertical part of vec_w at pt_x
        """
        # Start by ensuring vec_w is a tangent vector in the pre-shape space
        vec_w = super(ShapeSpace, self).to_tangent(pt_x, vec_w)
        # See equation (2) in Nava-Yazdani et al (2020), but note that all of our
        # equations are transposed from theirs
        xxT = pt_x.T @ pt_x
        wxT = vec_w.T @ pt_x
        return _solve_sylvester(xxT, xxT, wxT - wxT.T)

    def _vertical_tangent(self, pt_x: Point, vec_w: Vector) -> Vector:
        """The 'vertical' part of the tangent space is the part that doesn't count as
        movement in the quotient space, i.e. within equivalence classes. For example,
        north/south movement where equivalence = lines of longitude.

        The space of 'vertical' tangents, after accounting for shifts and scales with
        _aux_to_tangent, is the set of rotations. We get these by looking at the span
        of all 2D rotations – one per pair of axes in our space.
        """
        return pt_x @ self._solve_skew_symmetric_vertical_tangent(pt_x, vec_w)

    def inner_product(self, pt_x: Point, vec_w: Vector, vec_v: Vector):
        # Ensure that we're only measuring the 'horizontal' part of each tangent
        # vector. (We expect distance between two points to be equal to square root
        # norm of the logarithmic map between them).
        h_vec_w, h_vec_v = self._horizontal_tangent(pt_x, vec_w), self._horizontal_tangent(pt_x, vec_v)
        return super(ShapeSpace, self).inner_product(pt_x, h_vec_w, h_vec_v)

    def exp_map(self, pt_x: Point, vec_w: Vector) -> Point:
        # Decompose into horizontal and vertical parts. The vertical part specifies a
        # rotation in the sense that Skew-Symmetric matrices are the tangent space of
        # SO(p), and the vertical part equals Ax for some skew-symmetric matrix A. We
        # get from skew-symmetry to rotation using the matrix exponential,
        # i.e. rotation_matrix = matrix_exp(skew_symmetric_matrix)
        mat_a = self._solve_skew_symmetric_vertical_tangent(pt_x, vec_w)
        rotation = torch.matrix_exp(mat_a)
        horz_part = self._horizontal_tangent(pt_x, vec_w, vert_part=pt_x @ mat_a)
        # Apply vertical part, and note that rotation is equivariant with respect to
        # horizontal vectors, or horz_Rx(Rw)=Rhorz_x(w). This means that we (1)
        # rotate pt_x to pt_x', and (2) the new horizontal vector at pt_x' is equal
        # to the rotation applied to the original horizontal vector
        pt_x, horz_part = pt_x @ rotation, horz_part @ rotation
        # After applying the vertical part, delegate to the ambient PreShapeSpace for
        # the remaining horizontal part
        return super(ShapeSpace, self).exp_map(pt_x, horz_part)

    def log_map(self, pt_x: Point, pt_y: Point) -> Vector:
        # Only returns *horizontal* part of the tangent. Note that this means log_map
        # and exp_map are not inverses from the perspective of the PreShapeSpace,
        # but they are in the ShapeSpace. In other words, if c=exp_map(x,log_map(x,
        # y)), then we'll have length(y,c)=0 but not y==c Method: align y to x and
        # get x-->y' horizontal part from the PreShapeSpace's log_map
        _, new_b = _orthogonal_procrustes(pt_x, pt_y, anchor="x")
        return super(ShapeSpace, self).log_map(pt_x, new_b)

    def levi_civita(self, pt_x: Point, pt_y: Point, vec_w: Vector) -> Vector:
        # Both the horizontal and vertical parts of tangent vectors are equivariant
        # after rotation (Lemma 1b of Nava-Yazdani et al (2020)). This means we can
        # start by aligning x to y as follows to take care of the vertical part,
        # then all that's left is to transport the horizontal part:
        r_a, _ = _orthogonal_procrustes_rotation(pt_x, pt_y, anchor="y")
        new_pt_x, new_vec_w = pt_x @ r_a, vec_w @ r_a
        return super(ShapeSpace, self).levi_civita(new_pt_x, pt_y, new_vec_w)


def _orthogonal_procrustes_rotation(x, y, anchor="middle"):
    """Provided x and y, each matrix of size (m, p) that are already centered and
    scaled, solve the orthogonal procrustest problem (rotate x and y into a common
    frame that minimizes distances).

    If anchor="middle" (default) then both x and y
    If anchor="x", then x is left unchanged and y is rotated towards it
    If anchor="y", then y is left unchanged and x is rotated towards it

    See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :return: r_a and r_b, which, when right-multiplied with x and y, gives the
    aligned coordinates, or None for each if no transform is required
    """
    with torch.no_grad():
        u, _, v = torch.linalg.svd(x.T @ y)
    # Helpful trick to see how these are related: u is the inverse of u.T,
    # and likewise v is inverse of v.T. We get to the anchor=x and anchor=y solutions
    # by right-multiplying both return values by u.T or right-multiplying both return
    # values by v, respectively (if both return values are rotated in the same way,
    # it preserves the shape).
    if anchor == "middle":
        return u, v.T
    elif anchor == "x":
        return None, v.T @ u.T
    elif anchor == "y":
        return u @ v, None
    else:
        raise ValueError(f"Invalid 'anchor' argument: {anchor} (must be 'middle', 'x', or 'y')")


def _orthogonal_procrustes(x, y, anchor="middle"):
    """Provided x and y, each matrix of size (m, p) that are already centered and
    scaled, solve the orthogonal procrustest problem (rotate x and y into a common
    frame that minimizes distances).

    If anchor="middle" (default) then both x and y
    If anchor="x", then x is left unchanged and y is rotated towards it
    If anchor="y", then y is left unchanged and x is rotated towards it

    See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :return: new_a, new_b the rotated versions of x and y, minimizing element-wise
    squared differences
    """
    r_a, r_b = _orthogonal_procrustes_rotation(x, y, anchor)
    return x @ r_a if r_a is not None else x, y @ r_b if r_b is not None else y


def _solve_sylvester(x, y, q):
    # TODO - implement natively in pytorch so we don't have to convert to numpy on CPU and back again
    a_np, b_np, q_np = x.detach().cpu().numpy(), y.detach().cpu().numpy(), q.detach().cpu().numpy()
    return torch.tensor(solve_sylvester(a_np, b_np, q_np), dtype=x.dtype, device=x.device)
