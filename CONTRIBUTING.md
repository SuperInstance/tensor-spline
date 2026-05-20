# Contributing to tensor-spline

Thanks for your interest! Here's how to contribute:

## Development Setup

```bash
git clone <repo-url>
cd tensor-spline
pip install -e ".[mesh]"
```

## Running Tests

```bash
python -m pytest tensor_spline/tests/ -q --tb=short
```

## Guidelines

- **Bug fixes:** Open an issue first, then submit a PR referencing it.
- **Features:** Start a discussion before investing significant time.
- **Code style:** Follow the existing style. Docstrings use Google-style NumPy format.
- **Tests:** All new features and bug fixes must include tests.
- **Commits:** Write clear, descriptive commit messages.

## Reporting Issues

Include:
- Python / PyTorch version
- Minimal reproducible example
- Expected vs actual behavior
