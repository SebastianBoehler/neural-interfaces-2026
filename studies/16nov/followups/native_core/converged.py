"""Run §305 converged sklearn/native heads through the frozen §301 producer."""
import sys
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'expected exactly one §301 source anchor: {old[:100]}')
    return source.replace(old, new, 1)


def patch_source(source):
    source = replace_once(source, 'from bridge import Native  # noqa: E402',
                          'from bridge import Native, pointer\nfrom sklearn.linear_model import LogisticRegression')
    source = replace_once(source, "f'runs/s301_native_shards/{held}.npz'", "f'runs/s305_converged_shards/{held}.npz'")
    old = '''    started = time.monotonic()
    theta, loss, gradmax, calls = native.fit(standardized, y)
    if not np.isfinite([ref_loss, loss, gradmax]).all() or abs(loss - ref_loss) > 1e-5 or gradmax > 1e-5 or calls <= 0:
        raise ValueError(f'native head did not match the registered objective: {loss=} {ref_loss=} {gradmax=} {calls=}')
    coef = np.ascontiguousarray(theta[:, :-1]); intercept = np.ascontiguousarray(theta[:, -1])'''
    new = '''    started = time.monotonic()
    strict = LogisticRegression(C=.3, max_iter=10000, tol=1e-9).fit(standardized, y)
    strict_coef = np.ascontiguousarray(strict.coef_); strict_intercept = np.ascontiguousarray(strict.intercept_)
    strict_theta = np.ascontiguousarray(np.column_stack((strict_coef, strict_intercept)))
    strict_loss, strict_grad = native.objective(standardized, y, strict_theta)
    theta, loss, gradmax, calls = native.fit(standardized, y)
    strict_gradmax = float(np.max(np.abs(strict_grad)))
    if not np.isfinite([strict_loss, strict_gradmax, loss, gradmax]).all() or abs(loss - strict_loss) > 1e-5 or max(strict_gradmax, gradmax) > 1e-5 or calls <= 0:
        raise ValueError(f'converged objective parity failed: {loss=} {strict_loss=} {gradmax=} {strict_gradmax=} {calls=}')
    coef = np.ascontiguousarray(theta[:, :-1]); intercept = np.ascontiguousarray(theta[:, -1])
    def head(x, weights, bias):
        logits, probabilities = np.empty((len(x), 3), np.float64), np.empty((len(x), 3), np.float64)
        native.check(native.core.nc_head_predict(pointer(x), len(x), x.shape[1], pointer(fixed['mean']), pointer(fixed['scale']),
                     pointer(weights), pointer(bias), pointer(logits), pointer(probabilities)), 'nc_head_error')
        return logits, probabilities'''
    source = replace_once(source, old, new)
    source = replace_once(source, "    output = {'rows': saved['rows']}",
                          "    output = {'rows': saved['rows'], 'Nconverged_coef': coef, 'Nconverged_intercept': intercept,\n"
                          "              'Rconverged_coef': strict_coef, 'Rconverged_intercept': strict_intercept}")
    source = replace_once(source, '    member_differences = {}', '    member_differences = {}\n    solver_gaps = {}')
    old = "        p_train = native.predict(all_features, fixed['mean'], fixed['scale'], coef, intercept)"
    new = '''        z_train, p_train = head(all_features, coef, intercept)
        z_strict, p_strict = head(all_features, strict_coef, strict_intercept)
        probability_gap = float(np.max(np.abs(p_train - p_strict)))
        solver_gaps[order] = {'probability_max_abs': probability_gap, 'logit_max_abs': float(np.max(np.abs(z_train - z_strict)))}'''
    source = replace_once(source, old, new)
    source = replace_once(source, "        output[f'Ntrain|{order}'] = p_train - archive",
                          "        output[f'Nconverged|{order}'] = p_train - archive\n        output[f'Rconverged|{order}'] = p_strict - archive")
    source = replace_once(source, "                'native_fit_loss': loss, 'reference_loss': ref_loss, 'max_gradient': gradmax,",
        "                'native_fit_loss': loss, 'reference_loss': ref_loss, 'max_gradient': gradmax,\n"
        "                'strict_loss': strict_loss, 'strict_max_gradient': strict_gradmax, 'strict_iterations': int(strict.n_iter_.max()),\n"
        "                'strict_vs_native_coefficient_max_abs': float(np.max(np.abs(coef - strict_coef))),\n"
        "                'strict_vs_native_intercept_max_abs': float(np.max(np.abs(intercept - strict_intercept))),\n"
        "                'strict_vs_native_by_order': solver_gaps, 'converged_source_sha256': digest(ROOT / 'converged.sha256'),")
    return source


def main():
    path = Path(__file__).with_name('fit.py')
    exec(compile(patch_source(path.read_text()), str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})


if __name__ == '__main__':
    main()
