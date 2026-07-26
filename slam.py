#!/usr/bin/env python3
"""A pose-graph SLAM problem, generated and then solved.

Builds a robot trajectory, corrupts the odometry the way real odometry is
corrupted, adds loop closures, and poisons a fraction of them the way real
place recognition fails. Then measures how close the optimiser gets to a truth
it was never shown.

The interesting result is not that optimisation helps. It is what a handful of
wrong loop closures does to a least-squares estimate that has no defence
against them.
"""
from __future__ import annotations

import argparse
import math
import random

from graph import Kernel, PoseGraph
from se2 import SE2, exp, log, wrap_angle


def figure_eight(steps: int = 120, scale: float = 8.0) -> list[SE2]:
    """A trajectory that revisits its own middle, so loop closures are real."""
    poses = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        x = scale * math.sin(t)
        y = scale * math.sin(t) * math.cos(t)
        dx = scale * math.cos(t)
        dy = scale * (math.cos(t) ** 2 - math.sin(t) ** 2)
        poses.append(SE2(x, y, math.atan2(dy, dx)))
    return poses


def corridor_loop(steps: int = 100, size: float = 10.0) -> list[SE2]:
    """A square circuit returning to its start."""
    poses = []
    per_side = steps // 4
    for side in range(4):
        heading = side * math.pi / 2
        for i in range(per_side):
            progress = size * i / per_side
            corner = [(0, 0), (size, 0), (size, size), (0, size)][side]
            direction = [(1, 0), (0, 1), (-1, 0), (0, -1)][side]
            poses.append(SE2(corner[0] + direction[0] * progress,
                             corner[1] + direction[1] * progress,
                             wrap_angle(heading)))
    return poses


def build_problem(truth: list[SE2], seed: int = 1,
                  translation_noise: float = 0.05,
                  rotation_noise: float = 0.01,
                  loop_closures: int = 60,
                  outlier_fraction: float = 0.0,
                  min_separation: int = 12) -> tuple[PoseGraph, list[SE2], int]:
    """Odometry with drift, loop closures, and a chosen fraction of them wrong.

    Returns (graph, dead-reckoned initial guess, number of outliers planted).
    """
    rng = random.Random(seed)
    graph = PoseGraph()

    # Dead reckoning: integrate the noisy odometry, which is all a robot has
    # before any optimisation. Errors compound, so this drifts without bound.
    estimate = truth[0]
    guess = [estimate]
    measurements: list[SE2] = []

    for i in range(1, len(truth)):
        true_motion = truth[i - 1].between(truth[i])
        noisy = true_motion * exp((rng.gauss(0, translation_noise),
                                   rng.gauss(0, translation_noise),
                                   rng.gauss(0, rotation_noise)))
        measurements.append(noisy)
        estimate = estimate * noisy
        guess.append(estimate)

    for pose in guess:
        graph.add_pose(pose)
    graph.fix(0)

    odometry_information = [
        [1.0 / translation_noise ** 2, 0, 0],
        [0, 1.0 / translation_noise ** 2, 0],
        [0, 0, 1.0 / rotation_noise ** 2],
    ]
    for i, measurement in enumerate(measurements):
        graph.add_edge(i, i + 1, measurement, odometry_information, "odometry")

    # Loop closures between poses that are genuinely near each other in space
    # but far apart in time -- which is exactly when place recognition fires.
    candidates = [
        (i, j)
        for i in range(len(truth))
        for j in range(i + min_separation, len(truth))
        if math.dist(truth[i].translation, truth[j].translation) < 2.5
    ]
    rng.shuffle(candidates)
    chosen = candidates[:loop_closures]

    loop_information = [[100.0, 0, 0], [0, 100.0, 0], [0, 0, 400.0]]
    outliers = 0

    for i, j in chosen:
        if rng.random() < outlier_fraction:
            # A wrong match: the two places look alike but are not the same.
            # The measurement is confidently, completely incorrect.
            bogus = SE2(rng.uniform(-6, 6), rng.uniform(-6, 6),
                        rng.uniform(-math.pi, math.pi))
            graph.add_edge(i, j, bogus, loop_information, "outlier")
            outliers += 1
        else:
            true_relative = truth[i].between(truth[j])
            noisy = true_relative * exp((rng.gauss(0, 0.02), rng.gauss(0, 0.02),
                                         rng.gauss(0, 0.005)))
            graph.add_edge(i, j, noisy, loop_information, "loop")

    return graph, guess, outliers


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve a pose-graph SLAM problem.")
    ap.add_argument("--poses", type=int, default=100)
    ap.add_argument("--outliers", type=float, default=0.0,
                    help="fraction of loop closures that are wrong")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trajectory", choices=("figure-eight", "corridor"),
                    default="figure-eight")
    args = ap.parse_args()

    truth = (figure_eight(args.poses) if args.trajectory == "figure-eight"
             else corridor_loop(args.poses))

    print(f"\n  {args.trajectory}, {len(truth)} poses, "
          f"{args.outliers:.0%} of loop closures corrupted\n")

    results = []
    for kernel in (Kernel.NONE, Kernel.HUBER, Kernel.CAUCHY):
        graph, guess, outliers = build_problem(
            truth, seed=args.seed, outlier_fraction=args.outliers)

        if kernel is Kernel.NONE:
            initial_translation, initial_rotation = graph.absolute_error(truth)
            print(f"  {len(graph.edges)} edges "
                  f"({sum(1 for e in graph.edges if e.is_odometry)} odometry, "
                  f"{outliers} of the rest deliberately wrong)\n")
            print(f"  dead reckoning        "
                  f"{initial_translation:>8.3f} m   "
                  f"{math.degrees(initial_rotation):>7.2f} deg\n")

        report = graph.optimise(iterations=60, kernel=kernel, delta=1.0)
        translation, rotation = graph.absolute_error(truth)
        results.append((kernel, translation, rotation, report))

        label = {Kernel.NONE: "least squares", Kernel.HUBER: "Huber",
                 Kernel.CAUCHY: "Cauchy"}[kernel]
        print(f"  {label:<20} {translation:>8.3f} m   "
              f"{math.degrees(rotation):>7.2f} deg    "
              f"{report.iterations:>2} iters, chi2 "
              f"{report.initial_chi2:.3g} -> {report.final_chi2:.3g}")

    if args.outliers > 0:
        plain = next(r for r in results if r[0] is Kernel.NONE)
        best = min((r for r in results if r[0] is not Kernel.NONE),
                   key=lambda r: r[1])
        print(f"\n  With outliers present, the robust kernel is "
              f"{plain[1] / best[1]:.1f}x more accurate.")
        print("  Least squares has no way to disbelieve a measurement, so a few")
        print("  confident lies drag the whole map.\n")
    else:
        print("\n  With clean data every method agrees, which is the point:")
        print("  robustness costs nothing when there is nothing to be robust to.")
        print("  Run with --outliers 0.25 to see the difference.\n")


if __name__ == "__main__":
    main()
