#!/usr/bin/env python3
"""Pose graph optimisation: nonlinear least squares on a manifold.

The estimator behind SLAM back-ends and multi-sensor calibration. Odometry
gives you relative motion that drifts without bound; occasionally you recognise
somewhere you have been before, which says two poses far apart in time are
actually the same place. Reconciling all of it at once is the problem.

Three things make this harder than fitting a line:

    The state is not a vector      Poses live on SE(2). You cannot add an
                                   update to a heading and expect an answer.
                                   The optimiser works in a flat tangent space
                                   and folds results back onto the manifold.

    Jacobians are analytical       Numerical differentiation would work and
                                   would be forty times slower, and every
                                   analytical Jacobian is an opportunity for a
                                   sign error that produces slow convergence
                                   rather than an obvious failure. The tests
                                   check every one against finite differences.

    One bad loop closure ruins it  Least squares assumes Gaussian noise. A
                                   single wrong place-recognition match is
                                   thousands of sigma out, and the quadratic
                                   cost lets it drag the entire map. Robust
                                   kernels bound how much any one edge can
                                   pull.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from se2 import SE2, log, wrap_angle

Matrix = list[list[float]]
Vector = list[float]

# Identity information: measurements with no stated confidence get this.
DEFAULT_INFORMATION: Matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class Kernel(Enum):
    """How much a single wildly wrong measurement is allowed to matter."""

    NONE = "none"
    HUBER = "huber"
    CAUCHY = "cauchy"


def kernel_weight(kernel: Kernel, squared_error: float, delta: float) -> float:
    """Weight applied to an edge given its current squared error.

    Plain least squares weights everything equally, so an outlier at 100 sigma
    contributes 10,000 times a good measurement. Huber makes the cost linear
    beyond a threshold; Cauchy saturates it entirely.
    """
    if kernel is Kernel.NONE or squared_error <= 0:
        return 1.0
    if kernel is Kernel.HUBER:
        threshold = delta * delta
        if squared_error <= threshold:
            return 1.0
        return delta / math.sqrt(squared_error)
    # Cauchy
    return 1.0 / (1.0 + squared_error / (delta * delta))


@dataclass
class Edge:
    """A relative-pose measurement between two nodes."""

    source: int
    target: int
    measurement: SE2
    information: Matrix = field(default_factory=lambda: [r[:] for r in DEFAULT_INFORMATION])
    tag: str = ""

    @property
    def is_odometry(self) -> bool:
        return abs(self.target - self.source) == 1


@dataclass
class Report:
    iterations: int
    initial_chi2: float
    final_chi2: float
    converged: bool
    reason: str
    history: list[float] = field(default_factory=list)

    @property
    def reduction(self) -> float:
        if self.initial_chi2 <= 0:
            return 0.0
        return 1.0 - self.final_chi2 / self.initial_chi2

    def describe(self) -> str:
        return (f"  {self.iterations} iterations, chi2 "
                f"{self.initial_chi2:.4g} -> {self.final_chi2:.4g} "
                f"({self.reduction:.1%} reduction), {self.reason}")


class PoseGraph:
    def __init__(self) -> None:
        self.poses: list[SE2] = []
        self.edges: list[Edge] = []
        self.fixed: set[int] = set()

    # ---- building
    def add_pose(self, pose: SE2) -> int:
        self.poses.append(pose)
        return len(self.poses) - 1

    def add_edge(self, source: int, target: int, measurement: SE2,
                 information: Matrix | None = None, tag: str = "") -> Edge:
        for index in (source, target):
            if not 0 <= index < len(self.poses):
                raise ValueError(f"edge refers to unknown pose {index}")
        if source == target:
            raise ValueError("an edge cannot connect a pose to itself")
        edge = Edge(source, target, measurement,
                    information or [r[:] for r in DEFAULT_INFORMATION], tag)
        self.edges.append(edge)
        return edge

    def fix(self, index: int) -> None:
        """Anchor a pose.

        A pose graph made only of relative measurements has three unobservable
        degrees of freedom -- the whole map can be translated and rotated with
        no change in cost. Without an anchor the normal equations are singular
        and the solve fails, which is the correct behaviour but an unhelpful
        one, so anchoring is explicit.
        """
        if not 0 <= index < len(self.poses):
            raise ValueError(f"unknown pose {index}")
        self.fixed.add(index)

    # ---- residuals
    def residual(self, edge: Edge) -> Vector:
        """Error between what was measured and what the current estimate implies."""
        predicted = self.poses[edge.source].between(self.poses[edge.target])
        difference = edge.measurement.inverse() * predicted
        return [difference.x, difference.y, wrap_angle(difference.theta)]

    def jacobians(self, edge: Edge) -> tuple[Matrix, Matrix]:
        """Analytical derivatives of the residual with respect to both poses.

        Closed form rather than finite differences. Every entry is checked
        against numerical differentiation in the tests, because a wrong
        Jacobian does not crash -- it converges slowly, or to the wrong place,
        and looks like a tuning problem.
        """
        source, target = self.poses[edge.source], self.poses[edge.target]
        measurement = edge.measurement

        ci, si = math.cos(source.theta), math.sin(source.theta)
        cm, sm = math.cos(measurement.theta), math.sin(measurement.theta)

        # R_ij^T R_i^T, the rotation from the world frame into the measurement
        # frame.
        r_mt = [[cm, sm], [-sm, cm]]
        r_it = [[ci, si], [-si, ci]]
        combined = [
            [r_mt[0][0] * r_it[0][0] + r_mt[0][1] * r_it[1][0],
             r_mt[0][0] * r_it[0][1] + r_mt[0][1] * r_it[1][1]],
            [r_mt[1][0] * r_it[0][0] + r_mt[1][1] * r_it[1][0],
             r_mt[1][0] * r_it[0][1] + r_mt[1][1] * r_it[1][1]],
        ]

        dx = target.x - source.x
        dy = target.y - source.y

        # d(R_i^T)/d(theta_i) applied to the translation difference.
        d_rit = [[-si, ci], [-ci, -si]]
        rotated = [d_rit[0][0] * dx + d_rit[0][1] * dy,
                   d_rit[1][0] * dx + d_rit[1][1] * dy]
        theta_column = [r_mt[0][0] * rotated[0] + r_mt[0][1] * rotated[1],
                        r_mt[1][0] * rotated[0] + r_mt[1][1] * rotated[1]]

        jacobian_source = [
            [-combined[0][0], -combined[0][1], theta_column[0]],
            [-combined[1][0], -combined[1][1], theta_column[1]],
            [0.0, 0.0, -1.0],
        ]
        jacobian_target = [
            [combined[0][0], combined[0][1], 0.0],
            [combined[1][0], combined[1][1], 0.0],
            [0.0, 0.0, 1.0],
        ]
        return jacobian_source, jacobian_target

    def chi2(self, kernel: Kernel = Kernel.NONE, delta: float = 1.0) -> float:
        total = 0.0
        for edge in self.edges:
            error = self.residual(edge)
            squared = _weighted_norm(error, edge.information)
            if kernel is Kernel.NONE:
                total += squared
            elif kernel is Kernel.HUBER:
                threshold = delta * delta
                total += (squared if squared <= threshold
                          else 2 * delta * math.sqrt(squared) - threshold)
            else:
                total += delta * delta * math.log1p(squared / (delta * delta))
        return total

    def edge_errors(self, kernel: Kernel = Kernel.NONE) -> list[tuple[Edge, float]]:
        return sorted(
            ((e, _weighted_norm(self.residual(e), e.information)) for e in self.edges),
            key=lambda pair: -pair[1],
        )

    # ---- solving
    def optimise(self, iterations: int = 50, kernel: Kernel = Kernel.NONE,
                 delta: float = 1.0, tolerance: float = 1e-9,
                 initial_lambda: float = 1e-4) -> Report:
        """Levenberg-Marquardt.

        Gauss-Newton alone diverges when the initial guess is poor, which for a
        pose graph is normal -- odometry drift is exactly a poor initial guess.
        LM interpolates towards gradient descent when a step fails and back
        towards Gauss-Newton when it succeeds.
        """
        if not self.edges:
            raise ValueError("graph has no edges")
        if not self.fixed:
            raise ValueError(
                "no pose is fixed: a relative-only graph has three unobservable "
                "degrees of freedom and the normal equations are singular"
            )

        size = len(self.poses) * 3
        free = [i for i in range(len(self.poses)) if i not in self.fixed]
        if not free:
            return Report(0, self.chi2(kernel, delta), self.chi2(kernel, delta),
                          True, "every pose is fixed")

        index_of = {pose: position for position, pose in enumerate(free)}
        initial = self.chi2(kernel, delta)
        current = initial
        history = [current]
        lam = initial_lambda
        reason = "iteration limit reached"
        converged = False
        performed = 0

        for iteration in range(iterations):
            performed = iteration + 1
            hessian, gradient = self._build_system(free, index_of, kernel, delta)

            for _ in range(12):     # lambda search
                damped = [row[:] for row in hessian]
                for i in range(len(damped)):
                    damped[i][i] += lam * max(hessian[i][i], 1e-12)

                try:
                    step = _solve_spd(damped, [-g for g in gradient])
                except ValueError:
                    lam *= 10
                    continue

                backup = list(self.poses)
                self._apply(step, free)
                candidate = self.chi2(kernel, delta)

                if candidate < current:
                    current = candidate
                    lam = max(lam * 0.5, 1e-12)
                    break
                self.poses = backup
                lam *= 10
            else:
                reason = "damping could not find an improving step"
                converged = True
                break

            history.append(current)
            if len(history) > 1 and abs(history[-2] - history[-1]) < tolerance:
                reason = "converged"
                converged = True
                break

        return Report(performed, initial, current, converged, reason, history)

    def _build_system(self, free: list[int], index_of: dict[int, int],
                      kernel: Kernel, delta: float) -> tuple[Matrix, Vector]:
        size = len(free) * 3
        hessian = [[0.0] * size for _ in range(size)]
        gradient = [0.0] * size

        for edge in self.edges:
            error = self.residual(edge)
            weight = kernel_weight(kernel, _weighted_norm(error, edge.information), delta)
            js, jt = self.jacobians(edge)
            omega = edge.information

            blocks = []
            if edge.source in index_of:
                blocks.append((index_of[edge.source], js))
            if edge.target in index_of:
                blocks.append((index_of[edge.target], jt))

            for position, jacobian in blocks:
                # J^T Omega e
                jt_omega = _matmul(_transpose(jacobian), omega)
                contribution = _matvec(jt_omega, error)
                for i in range(3):
                    gradient[position * 3 + i] += weight * contribution[i]

            for position_a, jacobian_a in blocks:
                for position_b, jacobian_b in blocks:
                    block = _matmul(_matmul(_transpose(jacobian_a), omega), jacobian_b)
                    for i in range(3):
                        for j in range(3):
                            hessian[position_a * 3 + i][position_b * 3 + j] += \
                                weight * block[i][j]

        return hessian, gradient

    def _apply(self, step: Vector, free: list[int]) -> None:
        updated = list(self.poses)
        for position, pose_index in enumerate(free):
            pose = updated[pose_index]
            updated[pose_index] = SE2(
                pose.x + step[position * 3],
                pose.y + step[position * 3 + 1],
                wrap_angle(pose.theta + step[position * 3 + 2]),
            )
        self.poses = updated

    # ---- diagnostics
    def absolute_error(self, truth: list[SE2]) -> tuple[float, float]:
        """RMS translation and rotation error against a known trajectory."""
        if len(truth) != len(self.poses):
            raise ValueError("trajectory lengths differ")
        translation = sum(math.dist(p.translation, t.translation) ** 2
                          for p, t in zip(self.poses, truth))
        rotation = sum(wrap_angle(p.theta - t.theta) ** 2
                       for p, t in zip(self.poses, truth))
        n = len(truth)
        return math.sqrt(translation / n), math.sqrt(rotation / n)


# ------------------------------------------------------------- small linalg

def _transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def _matvec(a: Matrix, x: Vector) -> Vector:
    return [sum(row[j] * x[j] for j in range(len(x))) for row in a]


def _weighted_norm(error: Vector, information: Matrix) -> float:
    return sum(error[i] * information[i][j] * error[j]
               for i in range(3) for j in range(3))


def _solve_spd(a: Matrix, b: Vector) -> Vector:
    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 1e-14:
                    raise ValueError("system is not positive definite")
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]

    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
    return x
