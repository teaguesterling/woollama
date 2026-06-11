"""Package version, resolved from installed metadata.

`woollama` is a PEP 420 namespace package (no `__init__.py`, so the Rust
`woollama.core` wheel can share the namespace), which means there's no package
`__init__` to hold `__version__`. Callers import it from here instead.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("woollama")
except PackageNotFoundError:  # not installed (e.g. running from a raw checkout)
    __version__ = "0+unknown"
