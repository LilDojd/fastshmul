## fastshmul-py

This package uses [pyo3](https://pyo3.rs/) and
[maturin](https://github.com/PyO3/maturin) to bind fastmulsh functionality to
python as the `fastmulsh` package.

Recommended:

A clean python 3.10 environment with `maturin` installed. At which point running
`maturin develop` in this directory should build and install the package in the
environment. Run `pytest` in this directory to test everything is working.

Don't forget to use the `--release` flag when in production.
