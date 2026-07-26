#!/usr/bin/env python3
"""SE(2): rigid motions of the plane, as a Lie group.

A pose is not a vector. Adding two poses is meaningless, the difference between
two headings is not the subtraction of two numbers, and averaging 359 degrees
with 1 degree gives 180 unless you are careful. Every one of those is a real
bug in real robotics code.

The group operations here -- compose, invert, exp, log, adjoint -- are what let
an optimiser work in a flat tangent space while the state itself stays on the
manifold. The identities they satisfy are the tests: exp and log invert each
other, composition is associative, the adjoint relates the two.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Vector3 = tuple[float, float, float]
Matrix3 = list[list[float]]

# Below this rotation the small-angle series is both more accurate and better
# conditioned than the closed form, which divides by theta.
SMALL_ANGLE = 1e-8


def wrap_angle(theta: float) -> float:
    """Fold an angle into (-pi, pi].

    Every angular residual passes through here. Omitting it makes a loop
    closure that is off by a hair look like it is off by a full turn, and the
    optimiser will happily tear the map apart trying to fix it.
    """
    wrapped = math.fmod(theta + math.pi, 2 * math.pi)
    if wrapped <= 0:
        wrapped += 2 * math.pi
    return wrapped - math.pi


@dataclass(frozen=True)
class SE2:
    """A planar pose: translation (x, y) and heading theta."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    @classmethod
    def identity(cls) -> SE2:
        return cls(0.0, 0.0, 0.0)

    @property
    def normalised(self) -> SE2:
        return SE2(self.x, self.y, wrap_angle(self.theta))

    @property
    def translation(self) -> tuple[float, float]:
        return (self.x, self.y)

    def rotation(self) -> tuple[float, float]:
        return (math.cos(self.theta), math.sin(self.theta))

    def matrix(self) -> Matrix3:
        c, s = self.rotation()
        return [[c, -s, self.x], [s, c, self.y], [0.0, 0.0, 1.0]]

    def __mul__(self, other: SE2) -> SE2:
        """Compose: self then other, both read in self's frame."""
        c, s = self.rotation()
        return SE2(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            wrap_angle(self.theta + other.theta),
        )

    def inverse(self) -> SE2:
        c, s = self.rotation()
        return SE2(-c * self.x - s * self.y,
                   s * self.x - c * self.y,
                   wrap_angle(-self.theta))

    def between(self, other: SE2) -> SE2:
        """The motion from self to other, expressed in self's frame."""
        return self.inverse() * other

    def act(self, point: tuple[float, float]) -> tuple[float, float]:
        c, s = self.rotation()
        return (self.x + c * point[0] - s * point[1],
                self.y + s * point[0] + c * point[1])

    def adjoint(self) -> Matrix3:
        """Maps tangent vectors between the two frames of the group.

        Adj(T) v is the same motion v seen from the other side of T. It is what
        makes the chain rule work when a perturbation is applied on the far
        side of a composition.
        """
        c, s = self.rotation()
        return [[c, -s, self.y],
                [s, c, -self.x],
                [0.0, 0.0, 1.0]]

    def to_vector(self) -> Vector3:
        return (self.x, self.y, self.theta)

    @classmethod
    def from_vector(cls, v: Vector3) -> SE2:
        return cls(v[0], v[1], wrap_angle(v[2]))

    def __repr__(self) -> str:
        return (f"SE2(x={self.x:.4f}, y={self.y:.4f}, "
                f"theta={math.degrees(self.theta):.2f}deg)")


def exp(tangent: Vector3) -> SE2:
    """Tangent vector to group element.

    The translation part is not simply (vx, vy): moving forward while turning
    traces an arc, and V is the matrix that accounts for it. Treating exp as
    the identity on translation is a common shortcut that quietly biases every
    estimate with rotation in it.
    """
    vx, vy, omega = tangent

    if abs(omega) < SMALL_ANGLE:
        # Series expansion, exact to the order that matters here.
        return SE2(vx, vy, omega)

    sin_o, cos_o = math.sin(omega), math.cos(omega)
    a = sin_o / omega
    b = (1.0 - cos_o) / omega
    return SE2(a * vx - b * vy, b * vx + a * vy, wrap_angle(omega))


def log(pose: SE2) -> Vector3:
    """Group element to tangent vector. Inverse of exp."""
    omega = wrap_angle(pose.theta)

    if abs(omega) < SMALL_ANGLE:
        return (pose.x, pose.y, omega)

    half = omega / 2.0
    # cot(omega/2) * omega/2, written to stay stable as omega approaches zero.
    a = half / math.tan(half)
    v_inverse = [[a, half], [-half, a]]
    return (v_inverse[0][0] * pose.x + v_inverse[0][1] * pose.y,
            v_inverse[1][0] * pose.x + v_inverse[1][1] * pose.y,
            omega)


def interpolate(start: SE2, end: SE2, fraction: float) -> SE2:
    """Constant-velocity interpolation along the manifold.

    Interpolating x, y and theta independently cuts the corner and, near a
    heading wrap, goes the long way round.
    """
    return start * exp(tuple(fraction * c for c in log(start.between(end))))
