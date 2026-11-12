# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] - 2026-11-12

### Fixed
- Build failure with MSVC on Windows: `cpp/src/utils.cpp` used the alternative
  operator token `and`, which MSVC rejects by default. Replaced with `&&`.
  No behavior change on other platforms.

### Added
- Windows (AMD64) wheels are now built and published to PyPI.
- Windows added to the CI test matrix.

### Changed
- CI: macOS jobs moved from the retired `macos-14` runner to `macos-15`.
- CI: tests run on Python 3.13 via `actions/setup-python` instead of the
  runner's default Python in a venv.
- CI: CPython 3.15 skip widened from `cp315-*` to `cp315*` so free-threaded
  (`cp315t`) builds are skipped too; removed the redundant `pp*` skip
  (cibuildwheel v4 no longer builds PyPy by default).

## [0.9.0] - 2026-08-19

### Added
- `QuadcubicSpline`: cubic interpolation on a uniform 4D grid, matching the
  existing cubic/bicubic/tricubic API (`eval`, `deriv_*`, `coeff`, `coefficients`),
  including mixed partial derivatives.
- `QuadcubicSpline` w-direction derivatives on the Python class: `deriv_w`,
  `deriv_ww`, `deriv_wx`, `deriv_wy`, `deriv_wz` (previously only in the
  C++/Cython backend).
- `multispline.numba`: evaluate built splines inside `@njit` functions.
  - `@njit` kernels (`cubic_eval_d`, `bicubic_eval_d`, `tricubic_eval_d`,
    `quadcubic_eval_d`, plus convenience wrappers).
  - `jitclass` wrappers (`CubicSplineNumba` … `QuadcubicSplineNumba`) so
    `spline.eval(...)` / `spline.deriv_*(...)` work in nopython mode.
  - `to_numba()` on each spline class to build a Numba-ready view. The fit stays
    in C++; only evaluation runs under Numba.
- Optional install extra: `pip install multispline[numba]`.
- Test coverage for quadcubic and Numba paths; expanded tutorial.

### Fixed
- `TricubicSpline.deriv_zz` returned the second derivative in `y` instead of `z`
  (the Cython wrapper called `derivative_yy`). It now returns the correct `zz`
  derivative. **Behavior change:** results from `deriv_zz` on previous versions
  were incorrect and will differ.
- `SyntaxWarning: invalid escape sequence '\l'` emitted by the `coeff` docstrings
  on Python 3.12 (`\leq` → `\\leq`).

### Changed
- Refactored the C++ backend: the monolithic `cpp/src/spline.cpp` is split into
  `cubic.cpp`, `bicubic.cpp`, `tricubic.cpp`, `quadcubic.cpp`, and `utils.cpp`
  (adds `FourTensor`). No public API change.

### Notes
- Numba is an optional dependency; the core install is unchanged (`numpy` only).
- Requires Python >= 3.7 (3.9+ recommended).

[0.9.1]: https://github.com/znasipak/multispline/releases/tag/v0.9.1
