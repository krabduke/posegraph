"""Tests for SE(2) and pose-graph optimisation.

Three things are checked that a plausible-looking result would not reveal: the
group identities that make the manifold a manifold, every analytical Jacobian
against finite differences, and that the cost actually decreases monotonically
rather than merely ending up lower than it started.
"""
import math
import random

import pytest

from graph import Kernel, PoseGraph, kernel_weight
from se2 import SE2, exp, interpolate, log, wrap_angle
from slam import build_problem, corridor_loop, figure_eight


def random_pose(rng, span=5.0):
    return SE2(rng.uniform(-span, span), rng.uniform(-span, span),
               rng.uniform(-math.pi, math.pi))


def close(a: SE2, b: SE2, tol=1e-9):
    return (abs(a.x - b.x) < tol and abs(a.y - b.y) < tol
            and abs(wrap_angle(a.theta - b.theta)) < tol)


# ================================================================ angle wrap

@pytest.mark.parametrize("angle,expected", [
    (0.0, 0.0), (math.pi, math.pi), (-math.pi, math.pi),
    (2 * math.pi, 0.0), (3 * math.pi, math.pi), (1.5, 1.5),
])
def test_wrap_angle(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected)


def test_wrap_angle_lands_in_range():
    rng = random.Random(1)
    for _ in range(1000):
        assert -math.pi < wrap_angle(rng.uniform(-100, 100)) <= math.pi + 1e-12


def test_wrapping_handles_the_short_way_round():
    """359 degrees and 1 degree differ by 2, not 358. Every angular residual
    depends on this and getting it wrong tears a map apart."""
    a, b = math.radians(359), math.radians(1)
    assert abs(wrap_angle(b - a)) == pytest.approx(math.radians(2), abs=1e-9)


# ================================================================ group laws

def test_identity_is_neutral():
    rng = random.Random(2)
    for _ in range(100):
        pose = random_pose(rng)
        assert close(pose * SE2.identity(), pose)
        assert close(SE2.identity() * pose, pose)


def test_inverse_undoes_composition():
    rng = random.Random(3)
    for _ in range(200):
        pose = random_pose(rng)
        assert close(pose * pose.inverse(), SE2.identity())
        assert close(pose.inverse() * pose, SE2.identity())


def test_composition_is_associative():
    rng = random.Random(4)
    for _ in range(200):
        a, b, c = random_pose(rng), random_pose(rng), random_pose(rng)
        assert close((a * b) * c, a * (b * c), tol=1e-9)


def test_composition_is_not_commutative():
    """If it were, the implementation would be wrong: rotating then
    translating is not translating then rotating."""
    a, b = SE2(1, 0, 0), SE2(0, 0, math.pi / 2)
    assert not close(a * b, b * a)


def test_between_recovers_the_relative_motion():
    rng = random.Random(5)
    for _ in range(200):
        a, b = random_pose(rng), random_pose(rng)
        assert close(a * a.between(b), b)


def test_inverse_of_a_product_reverses_order():
    rng = random.Random(6)
    for _ in range(100):
        a, b = random_pose(rng), random_pose(rng)
        assert close((a * b).inverse(), b.inverse() * a.inverse())


def test_acting_on_a_point_matches_the_matrix():
    rng = random.Random(7)
    for _ in range(100):
        pose = random_pose(rng)
        point = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        m = pose.matrix()
        expected = (m[0][0] * point[0] + m[0][1] * point[1] + m[0][2],
                    m[1][0] * point[0] + m[1][1] * point[1] + m[1][2])
        assert pose.act(point) == pytest.approx(expected)


def test_composition_matches_matrix_multiplication():
    rng = random.Random(8)
    for _ in range(100):
        a, b = random_pose(rng), random_pose(rng)
        ma, mb = a.matrix(), b.matrix()
        product = [[sum(ma[i][k] * mb[k][j] for k in range(3)) for j in range(3)]
                   for i in range(3)]
        composed = (a * b).matrix()
        for i in range(3):
            for j in range(3):
                assert composed[i][j] == pytest.approx(product[i][j], abs=1e-12)


# ============================================================== exp and log

def test_exp_log_round_trip():
    rng = random.Random(9)
    for _ in range(1000):
        v = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-3.1, 3.1))
        assert log(exp(v)) == pytest.approx(v, abs=1e-9)


def test_log_exp_round_trip():
    rng = random.Random(10)
    for _ in range(500):
        pose = random_pose(rng)
        assert close(exp(log(pose)), pose, tol=1e-9)


def test_exp_of_zero_is_identity():
    assert close(exp((0.0, 0.0, 0.0)), SE2.identity())


def test_log_of_identity_is_zero():
    assert log(SE2.identity()) == pytest.approx((0.0, 0.0, 0.0))


def test_pure_translation_passes_through_unchanged():
    assert close(exp((2.0, 3.0, 0.0)), SE2(2.0, 3.0, 0.0))


def test_exp_of_rotation_traces_an_arc_not_a_straight_line():
    """Moving forward while turning does not end up straight ahead.

    Treating exp as the identity on translation is a common shortcut that
    quietly biases every estimate containing rotation.
    """
    result = exp((1.0, 0.0, math.pi / 2))
    assert result.x == pytest.approx(2 / math.pi, abs=1e-9)
    assert result.y == pytest.approx(2 / math.pi, abs=1e-9)
    assert result.x != pytest.approx(1.0)


def test_small_angle_branch_agrees_with_the_general_one():
    """The series and closed forms must meet at the switchover, or results jump
    discontinuously for slowly turning robots."""
    for omega in (1e-7, 1e-6, 1e-5, 1e-4):
        near = exp((1.0, 0.5, omega))
        below = exp((1.0, 0.5, omega / 10))
        assert abs(near.x - below.x) < 1e-4
        assert abs(near.y - below.y) < 1e-4


def test_exp_is_a_homomorphism_for_collinear_tangents():
    """exp(a v) exp(b v) = exp((a+b) v) along a single one-parameter subgroup."""
    v = (0.7, -0.3, 0.4)
    left = exp(tuple(0.3 * c for c in v)) * exp(tuple(0.5 * c for c in v))
    right = exp(tuple(0.8 * c for c in v))
    assert close(left, right, tol=1e-9)


def test_adjoint_relates_the_two_sides_of_a_composition():
    """T exp(v) = exp(Adj(T) v) T, the identity the chain rule depends on."""
    rng = random.Random(11)
    for _ in range(100):
        pose = random_pose(rng, span=2.0)
        v = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3))
        adjoint = pose.adjoint()
        transformed = tuple(sum(adjoint[i][j] * v[j] for j in range(3))
                            for i in range(3))
        assert close(pose * exp(v), exp(transformed) * pose, tol=1e-7)


def test_interpolation_hits_both_ends():
    rng = random.Random(12)
    a, b = random_pose(rng), random_pose(rng)
    assert close(interpolate(a, b, 0.0), a)
    assert close(interpolate(a, b, 1.0), b)


def test_interpolation_takes_the_short_way_round():
    a = SE2(0, 0, math.radians(-170))
    b = SE2(0, 0, math.radians(170))
    midpoint = interpolate(a, b, 0.5).theta
    assert abs(wrap_angle(midpoint - math.pi)) < 1e-9


# ================================================================ jacobians

def numerical_jacobian(graph, edge, which, eps=1e-7):
    columns = [[0.0] * 3 for _ in range(3)]
    for k in range(3):
        for sign in (1, -1):
            saved = graph.poses[which]
            delta = [0.0, 0.0, 0.0]
            delta[k] = sign * eps
            graph.poses[which] = SE2(saved.x + delta[0], saved.y + delta[1],
                                     saved.theta + delta[2])
            residual = graph.residual(edge)
            for i in range(3):
                columns[i][k] += sign * residual[i] / (2 * eps)
            graph.poses[which] = saved
    return columns


@pytest.mark.parametrize("seed", range(20))
def test_analytical_jacobians_match_finite_differences(seed):
    """The check that matters most.

    A wrong Jacobian does not crash. It converges slowly, or to the wrong
    answer, and looks like a tuning problem for as long as you let it.
    """
    rng = random.Random(seed)
    graph = PoseGraph()
    graph.add_pose(random_pose(rng, 3.0))
    graph.add_pose(random_pose(rng, 3.0))
    edge = graph.add_edge(0, 1, random_pose(rng, 3.0))

    analytic_source, analytic_target = graph.jacobians(edge)
    for which, analytic in ((0, analytic_source), (1, analytic_target)):
        numeric = numerical_jacobian(graph, edge, which)
        for i in range(3):
            for k in range(3):
                assert analytic[i][k] == pytest.approx(numeric[i][k], abs=1e-5), \
                    f"pose {which}, entry ({i},{k})"


def test_residual_is_zero_at_a_perfect_measurement():
    rng = random.Random(21)
    graph = PoseGraph()
    a, b = random_pose(rng), random_pose(rng)
    graph.add_pose(a)
    graph.add_pose(b)
    edge = graph.add_edge(0, 1, a.between(b))
    assert graph.residual(edge) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_residual_grows_with_disagreement():
    graph = PoseGraph()
    graph.add_pose(SE2.identity())
    graph.add_pose(SE2(1.0, 0.0, 0.0))
    small = graph.add_edge(0, 1, SE2(1.1, 0.0, 0.0))
    large = graph.add_edge(0, 1, SE2(3.0, 0.0, 0.0))
    assert (sum(abs(c) for c in graph.residual(large))
            > sum(abs(c) for c in graph.residual(small)))


# ================================================================== kernels

def test_kernel_none_never_downweights():
    assert kernel_weight(Kernel.NONE, 1e6, 1.0) == 1.0


def test_huber_is_unweighted_inside_its_threshold():
    assert kernel_weight(Kernel.HUBER, 0.5, 1.0) == 1.0


def test_huber_downweights_beyond_its_threshold():
    assert kernel_weight(Kernel.HUBER, 100.0, 1.0) == pytest.approx(0.1)


def test_cauchy_downweights_everything_progressively():
    near = kernel_weight(Kernel.CAUCHY, 1.0, 1.0)
    far = kernel_weight(Kernel.CAUCHY, 100.0, 1.0)
    assert 0 < far < near < 1.0


def test_weights_decrease_monotonically_with_error():
    for kernel in (Kernel.HUBER, Kernel.CAUCHY):
        weights = [kernel_weight(kernel, s, 1.0) for s in (0.1, 1, 10, 100, 1000)]
        assert all(a >= b for a, b in zip(weights, weights[1:]))


# ============================================================== optimisation

def test_optimising_a_perfect_graph_changes_nothing():
    rng = random.Random(30)
    truth = [random_pose(rng, 2.0) for _ in range(6)]
    graph = PoseGraph()
    for pose in truth:
        graph.add_pose(pose)
    for i in range(5):
        graph.add_edge(i, i + 1, truth[i].between(truth[i + 1]))
    graph.fix(0)

    report = graph.optimise()
    assert report.initial_chi2 == pytest.approx(0.0, abs=1e-16)
    for estimated, actual in zip(graph.poses, truth):
        assert close(estimated, actual, tol=1e-6)


def test_chi2_decreases_monotonically():
    """Levenberg-Marquardt rejects any step that does not improve the cost, so
    the history can never go up. A rise means the step acceptance is broken."""
    truth = corridor_loop(40)
    graph, _, _ = build_problem(truth, seed=5, loop_closures=20)
    report = graph.optimise(iterations=40)
    for earlier, later in zip(report.history, report.history[1:]):
        assert later <= earlier + 1e-12


def test_optimisation_recovers_ground_truth_it_never_saw():
    truth = figure_eight(60)
    graph, guess, _ = build_problem(truth, seed=11, loop_closures=40)

    before = graph.absolute_error(truth)[0]
    graph.optimise(iterations=60)
    after = graph.absolute_error(truth)[0]

    assert after < before / 2
    assert after < 0.5


def test_dead_reckoning_drifts_without_loop_closures():
    truth = figure_eight(60)
    graph, _, _ = build_problem(truth, seed=12, loop_closures=0)
    drift = graph.absolute_error(truth)[0]
    assert drift > 0.1


def test_more_loop_closures_help():
    truth = figure_eight(60)
    errors = []
    for closures in (0, 10, 40):
        graph, _, _ = build_problem(truth, seed=13, loop_closures=closures)
        graph.optimise(iterations=60)
        errors.append(graph.absolute_error(truth)[0])
    assert errors[0] > errors[2]


def test_unfixed_graph_is_rejected():
    graph = PoseGraph()
    graph.add_pose(SE2.identity())
    graph.add_pose(SE2(1, 0, 0))
    graph.add_edge(0, 1, SE2(1, 0, 0))
    with pytest.raises(ValueError, match="unobservable"):
        graph.optimise()


def test_graph_without_edges_is_rejected():
    graph = PoseGraph()
    graph.add_pose(SE2.identity())
    graph.fix(0)
    with pytest.raises(ValueError, match="no edges"):
        graph.optimise()


def test_self_loop_rejected():
    graph = PoseGraph()
    graph.add_pose(SE2.identity())
    with pytest.raises(ValueError, match="itself"):
        graph.add_edge(0, 0, SE2.identity())


def test_edge_to_unknown_pose_rejected():
    graph = PoseGraph()
    graph.add_pose(SE2.identity())
    with pytest.raises(ValueError, match="unknown pose"):
        graph.add_edge(0, 5, SE2.identity())


def test_fixed_pose_does_not_move():
    truth = corridor_loop(40)
    graph, _, _ = build_problem(truth, seed=14, loop_closures=15)
    anchor = graph.poses[0]
    graph.optimise(iterations=30)
    assert close(graph.poses[0], anchor, tol=1e-12)


# ============================================================== robustness

def test_outliers_wreck_plain_least_squares():
    """The reason robust kernels exist.

    Least squares has no way to disbelieve a measurement: cost grows with the
    square of the error, so a confidently wrong loop closure outvotes many
    correct ones.
    """
    truth = figure_eight(100)

    plain, _, planted = build_problem(truth, seed=7, outlier_fraction=0.30,
                                      loop_closures=60, min_separation=12)
    assert planted > 0
    plain.optimise(iterations=80, kernel=Kernel.NONE)
    plain_error = plain.absolute_error(truth)[0]

    robust, _, _ = build_problem(truth, seed=7, outlier_fraction=0.30,
                                 loop_closures=60, min_separation=12)
    robust.optimise(iterations=80, kernel=Kernel.HUBER, delta=1.0)
    robust_error = robust.absolute_error(truth)[0]

    assert plain_error > robust_error * 3


def test_robust_kernels_cost_nothing_on_clean_data():
    truth = figure_eight(100)
    errors = {}
    for kernel in (Kernel.NONE, Kernel.HUBER, Kernel.CAUCHY):
        graph, _, planted = build_problem(truth, seed=7, outlier_fraction=0.0,
                                          loop_closures=60, min_separation=12)
        assert planted == 0
        graph.optimise(iterations=80, kernel=kernel)
        errors[kernel] = graph.absolute_error(truth)[0]

    baseline = errors[Kernel.NONE]
    for kernel, error in errors.items():
        assert error == pytest.approx(baseline, rel=0.15), kernel


def test_outlier_edges_end_up_with_the_largest_residuals():
    truth = figure_eight(100)
    graph, _, planted = build_problem(truth, seed=7, outlier_fraction=0.30,
                                      loop_closures=60, min_separation=12)
    graph.optimise(iterations=80, kernel=Kernel.CAUCHY)

    ranked = graph.edge_errors()
    worst = [edge for edge, _ in ranked[:planted]]
    caught = sum(1 for edge in worst if edge.tag == "outlier")
    assert caught >= planted * 0.6


def test_absolute_error_rejects_a_mismatched_trajectory():
    graph = PoseGraph()
    graph.add_pose(SE2.identity())
    with pytest.raises(ValueError, match="lengths differ"):
        graph.absolute_error([SE2.identity(), SE2.identity()])
