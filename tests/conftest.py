"""The --backend option and the compiled scenes the suite shares.

hpcBilliards is not installed -- there is no pyproject.toml and no setup.py --
so it imports as a top-level package from the repo root and nowhere else. The
sys.path line below is what lets `pytest` run from any directory.

Nothing here imports matplotlib. Importing pyplot before compileScene makes an
OpenMP target region fall back to the host silently, which CompiledScene raises
on for the gpu backend; no test needs a figure, so the trap is simply avoided.
"""

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpcBilliards import compileScene            # noqa: E402
from hpcBilliards.compile import BACKENDS        # noqa: E402

import helpers                                   # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--backend", default="linear", choices=sorted(BACKENDS),
        help="which backend to build the scenes with. linear needs only g++, "
             "which is why it is the default; openmp is what the library "
             "itself defaults to, and gpu_openmp needs a device to offload to")


@pytest.fixture(scope="session")
def backend(request):
    return request.config.getoption("--backend")


# One build per distinct STRUCTURE, which is all the cache key is: a scene whose
# numbers differ compiles to the same .so and dlopen returns the same library.
# So these hand back a compiled scene, not a configured one -- a test sets the
# parameters it wants with helpers.push() and does not inherit the last test's.
#
# Session-scoped because a cold build costs a second or two and a warm one is a
# cache hit; the fixtures below are roughly five builds for the whole suite.


def _compiled(scene, backend):
    return compileScene(scene, backend=backend)


@pytest.fixture(scope="session")
def squareBox(backend):
    """four lineSegments: the closed unit box"""
    return _compiled(helpers.squareScene(g=0.0), backend)


@pytest.fixture(scope="session")
def roundBilliard(backend):
    """one elipse"""
    return _compiled(helpers.circleScene(g=1.0), backend)


@pytest.fixture(scope="session")
def portalChannel(backend):
    """two lines and a portal. The flips and both faces are parameters, so every
    orientation this suite tests comes out of this one build."""
    return _compiled(helpers.portalScene(), backend)


@pytest.fixture(scope="session")
def basinBox(backend):
    """three walls and a basin: the only structure here that getBasins accepts"""
    return _compiled(helpers.basinScene(g=1.0), backend)


@pytest.fixture(scope="session")
def oneWall(backend):
    """a single line. Shared by the free-flight window and the vertical throw --
    same structure, same library, different numbers."""
    return _compiled(helpers.freeFlightScene(), backend)
