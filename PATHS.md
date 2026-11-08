# Finite deletion paths

## Change weights, not the objective convention

`trace_deletion(X,y,query,label,indices,ridge=.1,segments=16)` starts with the same all-one weights as `audit`. For a selected group G, define `w_i(t)=1-t` if i is in G, otherwise 1. The objective always uses the original row count N and the same positive ridge. Every path point is refitted from zero with the existing convergence requirement, not predicted by the base influence vector. Damping is deliberately absent.

For each t, the stationarity condition is `gradient R(theta(t),w(t))=0`. Differentiating it gives

`H(t) theta'(t) = sum_{i in G} gradient loss_i(theta(t)) / N`.

The positive sign comes from `w'_i(t)=-1`. All loss gradients and the Hessian are evaluated at the *current* refitted theta. For the query loss, `f'(t)=gradient query_loss(theta(t))^T theta'(t)`.

Reports preserve every t, weight, full fit/Hessian/Cholesky/history, selected sample gradient, right-hand side, parameter derivative, query derivative and linear-solve residual. No unconverged fit is accepted. Canonical sorted unique row indices make selection order irrelevant; rows and original row indices are never silently merged.

At t=0, linearity of the Hessian solve means the group derivative is the sum of the original individual deletion predictions. At t=1, group deletion is generally **not** the sum of separately refitted individual deletions. Those endpoints follow different nonlinear parameter paths.

## Integrate, then compare with an independent endpoint

For a smooth exact optimum path, the fundamental theorem of calculus gives `f(1)-f(0)=integral_0^1 f'(t) dt`. Positive ridge supplies a unique strongly convex optimum throughout this nonnegative-weight path. In floating point, neither refits nor quadrature are exact.

The implementation fits a uniform fine grid `t=k/S`, for `S` a power of two from 2 through 64. Composite Simpson uses endpoint coefficients 1, alternating interior coefficients 4 and 2, and divides the weighted sum by `3S`. Coarse grids S=2,4,... reuse the same fine-grid fits, not additional training. Composite trapezoid estimates are also retained as a separate numerical comparison. No SciPy dependency is used; its [Simpson documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html) describes the standard rule.

Each level reports Simpson/trapezoid estimates, their signed gaps against the directly measured endpoint loss difference, and the Simpson difference from the previous level. A small successive difference is **not** labeled a rigorous error bound. Error need not shrink monotonically on arbitrary accepted inputs; the code does not silently adapt until a chosen answer looks good. A fixed grid can miss sharp curvature. The endpoint refit is included so the approximation gap is actually measurable in this small reference implementation.

## Developed fixture and controls

The same seven-row fixture and query from the original demo are used, with original N=7 and ridge 0.1. Each path has 17 converged fits at S=16; the three paths total 51. The coarse quadrature levels reuse those fits.

| Selected group | Frozen first-order loss delta | Direct endpoint loss delta | Simpson 16 | Simpson minus endpoint |
|---|---:|---:|---:|---:|
| row 5 | +0.203204263 | +0.305655211 | +0.305655349 | +1.381e-7 |
| rows5,6 | -0.149821907 | -0.357506965 | -0.357502140 | +4.825e-6 |
| all 7 rows | -0.004253150 | +0.079105273 | +0.079849431 | +7.442e-4 |

Deleting all rows reverses the sign of the frozen first-order prediction on this developed fixture. The endpoint has theta exactly zero in the computation because only ridge remains; its query loss is log(2). The final path derivative is `15/14`, so the visible steep tail is not discarded. Simpson 16 still has a gap about 0.000744, larger than the other two cases; it remains visible rather than being gated away. The fine-grid polyline connects the actual derivative samples, not a fitted smooth curve.

Jointly deleting rows 5 and 6 has endpoint change `-0.357506965`. The sum of the two *individual* numerical deletion changes is `-0.142898169`, differing by about `-0.214608797`. That is finite group nonadditivity, not a causal interaction effect. The group first-order derivative is still the sum of the original local derivatives, as the test checks.

## Use and verification

```sh
python -m influencelab path-demo > path-report.json
python -m influencelab trace examples/path-input.json > one-path.json
python examples/render_paths.py --check
```

The strict version-1 trace request has exactly `version`, `X`, `y`, `query`, `label`, `ridge`, `indices`, `segments`. It inherits original feature/label/ridge/file limits. Indices must be a nonempty unique list of in-range integers, not booleans or floats. Segments must be an integer power of two in 2..64. There is no retained-row-mean path here; this path always keeps original N fixed.

Seventeen new tests supplement the original 26: 14 behavior and 3 SVG tests. Behavior checks include independent polynomial integrals of degrees 0-3, all path weights/convergence/Hessian equations, interior parameter and query derivatives against separately perturbed refits, initial individual-sum identities, actual grouped endpoint refits, all-row sign reversal and nonzero integration error, finite group nonadditivity, row-selection order, input immutability/RNG, permutation/orthogonal coordinates, single/all-zero-feature cases, strict CLI/API and both committed outputs. A fit-call-count test confirms 65 fits on a 64-segment path, reused across all 6 quadrature levels. SVG tests check every summary/convergence cell, every point of all 3 polylines, complete compressed metadata with a hidden interior-Hessian tamper and byte-exact regeneration.

Computational work is S+1 full convex fits plus dense derivative solves, not a scalable way to avoid retraining. The full endpoint is available precisely because the examples are small. This is a numerical reference for local versus finite sensitivity, not a data-deletion service, machine-unlearning algorithm, privacy proof, statistical confidence statement or causal attribution. No claim that finite deletion must follow the base derivative's sign.
