# InfluenceLab fit and influence method

## One objective, one weight convention

For supplied rows `x_i`, binary integer labels `y_i`, sample weights `w_i` and positive ridge `lambda`, minimize

`R(theta,w) = (1 / N) sum_i w_i [softplus(theta.x_i) - y_i theta.x_i] + lambda ||theta||^2 / 2`.

The implementation evaluates the equivalent signed-softplus expression to avoid cancellation. At the audit's base fit, all weights are 1 and `N` is the original row count. No intercept is inserted; every supplied coordinate is regularized equally. Orthogonal changes of coordinates preserve this ridge objective; arbitrary feature rescaling does not, so no general parameterization-invariance claim is made.

`g_i = (sigmoid(z_i)-y_i) x_i` and `H = sum_i w_i p_i(1-p_i) x_i x_i^T / N + lambda I`.

For any real vector `v`, `v^T H v = sum_i w_i p_i(1-p_i)(x_i.v)^2/N + lambda ||v||^2`. With nonnegative weights and positive ridge, the exact mathematical Hessian is positive definite even with duplicate or zero features. Floating-point Cholesky still checks positive pivots and refuses failure; it adds no hidden jitter.

At a converged optimum, implicit differentiation gives

`D_i = d theta / d w_i = -H^-1 g_i / N`.

The query-loss derivative is `gradient(query_loss)^T D_i`. Reports independently evaluate it with the adjoint solve `s = H^-1 gradient(query_loss)` as `-s^T g_i/N`. Each parameter solve includes every residual component. A positive derivative means increasing the row's weight increases this query's loss locally; it does not establish that the row is incorrect or globally harmful.

Koh and Liang's additive external epsilon convention perturbs the mean loss by `epsilon * loss_i`. Here `w_i` multiplies a loss *inside* the original-N mean, so `epsilon = delta w_i/N`. Do not drop the factor of `1/N` or reverse the deletion sign.

## Finite deletion and normalization

Deleting one row means changing its weight from 1 to 0. The linear parameter prediction is `-D_i`; the linear query-loss change is `-gradient(query_loss)^T D_i`. The actual query-loss change also includes both nonlinear parameter motion and nonlinear query loss. Neither is presumed infinitesimal.

The fixed-N numerical reference sets `w_i=0`, keeps N and lambda fixed and refits from zero to the same convergence criterion. The retained-mean reference instead divides by `N-1` with the *same* lambda. It is a distinct objective. Multiplying the fixed-N objective by the positive scalar `N/(N-1)` shows that fixed-N deletion equals a retained-mean fit with ridge `lambda*N/(N-1)`, not with ridge lambda. The test verifies that equivalence. For a single-row dataset, deletion leaves ridge alone with theta zero; a retained-row mean is undefined and omitted.

## Damping is an explicitly different inverse

Optional delta replaces the matrix in the influence solve by `H + delta I` at the same base optimum. For delta greater than zero, the report sets `is_exact_undamped_derivative=false`. The base fit and deletion references remain unchanged. A model refitted with lambda increased by delta generally changes theta, p, H and the query gradient, so it is not the same operation. Damping can change influence rankings and accuracy; no monotonic finite-deletion improvement is promised.

## Solver

Start theta at zero. Compute stable signed-logistic losses, full gradients and dense Hessians. Use an unjittered Cholesky Newton direction and Armijo backtracking with coefficient 1e-4, at most 60 halvings. At objective roundoff, accept within eight ulps of the current objective only if the max absolute gradient strictly decreases. The history records whether that exception was used and the accepted step scale. Default gradient tolerance is 1e-10; failure to converge within 100 steps raises an error. No unconverged iterate is passed to influence.

Strong convexity gives `||theta - theta_star||_2 <= ||gradient R(theta)||_2 / lambda` in exact mathematical evaluation. The reported floating-point bound uses the computed gradient and is a useful numerical diagnostic, not an interval with directed-rounding certification. The solver is numerical, not a symbolic exact optimizer.

For large logits, sigmoid and its label-specific residual are evaluated without subtracting nearly equal numbers. Curvature uses `exp(-abs(z))/(1+exp(-abs(z)))^2`, retaining small positive curvature beyond the point where rounded sigmoid becomes 1. At extreme magnitudes it may underflow to zero; positive ridge still remains. Features and work limits keep ordinary arithmetic bounded. No arbitrary-precision or extreme-condition-number guarantee.

## Inputs and cost

- 1-128 rows, 1-12 explicit feature columns, each finite numeric value in [-100,100]. Booleans are not numbers. Labels are integer 0 or 1, not floats or booleans.
- Ridge in [1e-6,100], damping in [0,100], weights in [0,10]. Normalizer defaults to row count and supports [1,128]. Tolerance [1e-12,1e-6], integer max steps 1-200.
- Strict version-1 JSON request requires exactly `version`, `X`, `y`, `query`, `label`, `ridge`, `damping`, `refit`. Duplicate keys, nonfinite values, unknown keys, bad shapes and files above 100000 bytes fail with exit 2. No best-effort row drops.
- Fitting costs roughly O(iterations * (N d^2 + d^3)), with bounded backtracking. The dense influence API solves once per row plus the adjoint; optional refits perform up to 2N additional fits. No speedup benchmark or scalable inverse-Hessian approximation is claimed.

All work is offline with deterministic arithmetic order. Inputs are copied, no random global state is read or changed, and no raw supplied row is sent to a network. Do not put private training data into public example files.

## Independent checks and limits

The suite checks primitive logistic derivatives, all objective derivatives, inverse solves, a separate scalar bisection optimizer, and every weight direction against independently perturbed converged refits on the visible fixture and three seeded systems through six parameters. A developed stress fixture exercises safeguarded backtracking. The local remainder experiment checks quadratic error decrease at three small negative weight changes for row 5, not a universal rate theorem for all accepted inputs.

All training examples and queries are supplied, small and synthetic. There is no statistical sampling assumption, effect-identification claim, privacy mechanism or trained deep model. The displayed finite-deletion error is evidence of an approximation limit, not a failure hidden by a gate. Attribution and source links are in README.
