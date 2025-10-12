"""Sample code module for ingestion testing."""

from __future__ import annotations


def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using simple iteration."""

    if n < 2:
        return max(n, 0)
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


if __name__ == "__main__":  # pragma: no cover
    import sys

    value = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(fibonacci(value))
