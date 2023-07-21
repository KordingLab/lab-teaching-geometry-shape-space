import enum
import torch
from typing import Callable, Tuple
from manifold import Point, Scalar, Manifold


class OptimResult(enum.Enum):
    CONVERGED = 0
    MAX_STEPS_REACHED = 1
    CONDITIONS_VIOLATED = 2
    ILL_POSED = 3
    NO_OPT_NEEDED = 0


def minimize(manifold: Manifold,
             fun: Callable[[Point], Scalar],
             init: Point,
             *,
             pt_tol: float = 1e-6,
             fn_tol: float = 1e-6,
             init_step_size: float = 0.1,
             max_step_size: float = 100.,
             wolfe_c1=1e-4,
             wolfe_c2=0.9,
             wolfe_c2_min=1e-2,
             max_iter: int = 10000) -> Tuple[Point, OptimResult]:
    """Function minimization on a Length Space by gradient descent and line search.

    :param manifold: manifold on which to perform optimization
    :param fun: torch-differentiable function to be minimized
    :param init: initial point
    :param pt_tol: convergence tolerance for changes in the coordinate
    :param fn_tol: convergence tolerance for changes in the function
    :param init_step_size: initial gradient descent step size
    :param max_step_size: largest sane gradient descent step size
    :param wolfe_c1: threshold on first Wolfe condition (progress check - is function improving at least a little?)
    :param wolfe_c2: threshold on second Wolfe condition (curvature check - is gradient changing by not too much?)
    :param wolfe_c2_min: rarely both conditions fail, so we reduce c2. Stop trying to do thise when wolfe_c2 < wolfe_c2_min
    :param max_iter: break regardless of convergence if this many steps reached
    :return: tuple of (point, OptimResult) where the OptimResult indicates convergence status
    """

    step_size, pt = init_step_size, init.clone()
    fval, grad = fun(pt), torch.autograd.functional.jacobian(fun, pt)
    for itr in range(max_iter):
        # Update by gradient descent + line search
        step_direction = -1 * grad
        new_pt = manifold.project(pt + step_size * step_direction)
        new_fval, new_grad = fun(new_pt), torch.autograd.functional.jacobian(fun, new_pt)

        # Test for convergence
        if manifold.distance(pt, new_pt) <= pt_tol and \
                new_fval <= fval and \
                fval - new_fval <= fn_tol:
            return new_pt if new_fval < fval else pt, OptimResult.CONVERGED

        # Check Wolfe conditions
        sq_step_size = (grad * step_direction).sum()
        condition_i = new_fval <= fval + wolfe_c1 * step_size * sq_step_size
        condition_ii = (step_direction*new_grad).sum() >= wolfe_c2 * sq_step_size
        if condition_i and condition_ii:
            # Both conditions met! Update pt and loop.
            pt, fval, grad = new_pt.clone(), new_fval, new_grad
        elif condition_i and not condition_ii:
            # Step size is too small - accept the new value but adjust step_size for next loop
            pt, fval, grad = new_pt.clone(), new_fval, new_grad
            step_size = min(max_step_size, step_size * 1.1)
        elif condition_ii and not condition_i:
            # Step size is too big - adjust and loop, leaving pt, fval, and grad unchanged
            step_size *= 0.8
        else:
            # Both conditions violated, indicating that the curvature is high (so condition 2 fails) and the function is
            # barely changing (so condition 1 fails). When this first happens, we can make some more (slow) progress by
            # making the threshold on c2 less strict. But eventually we will give up when wolfe_c2 < wolfe_c2_min
            wolfe_c2 *= 0.8
            if wolfe_c2 < wolfe_c2_min:
                return pt, OptimResult.CONDITIONS_VIOLATED
            elif new_fval < fval:
                # Despite condition weirdness, the new fval still improved. Accept the new point then loop.
                pt, fval, grad = new_pt.clone(), new_fval, new_grad

    # Max iterations reached – return final value of 'pt' with flag indicating max steps reached
    return pt, OptimResult.MAX_STEPS_REACHED
