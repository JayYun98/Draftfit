"""Source-only research tests. Run the suite with python -m draftfit.research.tests.cpu."""


def load_tests(loader, tests, pattern):
    # unittest calls this hook instead of recursing during repository discovery.
    return tests
