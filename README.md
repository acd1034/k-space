# kspc: k-space calculation
<!--
[![Linux build status](https://github.com/acd1034/iris/actions/workflows/linux-build.yml/badge.svg)](https://github.com/acd1034/iris/actions/workflows/linux-build.yml)
[![macOS build status](https://github.com/acd1034/iris/actions/workflows/macos-build.yml/badge.svg)](https://github.com/acd1034/iris/actions/workflows/macos-build.yml)
-->

Click [here](https://acd1034.github.io/k-space/index.html) to see the HTML documentation generated by Doxygen.

## Noteworthy Features

## Supported Compilers
The code will work on the following compilers:
- GCC (latest)
- Clang (latest)
- Apple clang (version 11.0.0 or later)

## Library Dependencies
- `<kspc/core.hpp>`, `<kspc/math.hpp>`, `<kspc/kspc.hpp>` → depend on no external library
- `<kspc/integration.hpp>` → `GSL`
- `<kspc/linalg.hpp>` → `BLAS`, `LAPACK`
