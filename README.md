# posegraph

Pose graph optimisation: nonlinear least squares on a manifold. The estimator
behind SLAM back-ends and multi-sensor calibration.

```
$ python3 slam.py --outliers 0.25

  dead reckoning           0.934 m      2.37 deg

  least squares            0.811 m      6.33 deg
  Huber                    0.221 m      0.75 deg
  Cauchy                   0.229 m      0.80 deg

  With outliers present, the robust kernel is 3.7x more accurate.
```

With clean data every method agrees to three decimal places. Robustness costs
nothing when there is nothing to be robust to — and is the difference between a
map and a mess when there is.

## Three things that make this harder than fitting a line

**A pose is not a vector.** Adding two poses is meaningless, and averaging 359°
with 1° gives 180° unless you are careful. SE(2) is implemented as a proper Lie
group — compose, invert, exp, log, adjoint — and the identities are the tests:
exp and log invert each other, composition is associative, and
`T exp(v) = exp(Adj(T) v) T` holds to 1e-7.

`exp` is not the identity on translation. Moving forward while turning traces
an arc: `exp((1, 0, π/2))` lands at `(2/π, 2/π)`, not `(1, 0)`. Treating it as
a straight line quietly biases every estimate containing rotation, and there is
a test pinning the exact value.

**Jacobians are analytical.** Closed form rather than finite differences,
because a wrong Jacobian does not crash — it converges slowly, or to the wrong
place, and looks like a tuning problem for as long as you let it. Every entry
is checked against numerical differentiation across 20 random configurations,
agreeing to 1e-9.

**One bad loop closure ruins everything.** Least squares has no way to
disbelieve a measurement: cost grows with the square of the error, so a
confidently wrong place-recognition match at 60 sigma outvotes many correct
ones. Huber and Cauchy kernels bound how much any single edge can pull.

| outlier rate | least squares | Huber | Cauchy |
|---|---|---|---|
| 0% | 0.175 m | 0.175 m | 0.177 m |
| 10% | 1.016 m | 0.168 m | 0.168 m |
| 30% | 1.498 m | 0.152 m | 0.168 m |

## Solver

Levenberg-Marquardt, because Gauss-Newton diverges from a poor initial guess
and odometry drift *is* a poor initial guess. Any step that fails to reduce the
cost is rejected and the damping increased, so **χ² decreases monotonically** —
asserted over the whole history, not just start to finish.

A relative-only graph has three unobservable degrees of freedom: the entire map
can be translated and rotated at no cost. Anchoring is explicit, and optimising
without it raises rather than silently returning one of infinitely many
answers.

Stdlib only. Tests: `python3 -m pytest test_posegraph.py` (68 tests)

## Not handled

**SE(2) only** — no SE(3), so this is planar robots and not drones in flight.
The update is additive on (x, y, θ) with angle wrapping rather than a true
exponential retraction; that is the standard SE(2) pose-graph parameterisation
and what g2o's SE2 vertex does, but it is a distinction worth naming.

Dense Cholesky on the full information matrix, which ignores the sparsity that
makes real SLAM tractable — fine to a couple of hundred poses, hopeless at ten
thousand. No landmarks, no switchable constraints, no marginalisation, no
covariance recovery.
