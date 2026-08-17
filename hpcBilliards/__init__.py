"""AnalyticBilliard as a shared library.

    from hpcBilliards import compileScene, updateScene

    compiled = compileScene(SCENE, backend="openmp")
    rec = compiled.recordWithIterations(x, y, vx, vy, iterations=1000)
    ser = compiled.recordWithTime(x, y, vx, vy, tf=100.0, dt=0.01)

Two ways to record the same run: recordWithIterations gives one row per event,
recordWithTime one row per tick of a uniform grid.

Declare the objects once and compile; after that, parameters, the portal flips,
gravity and deadTime all change without touching the compiler. Only adding,
removing or retyping an object needs a rebuild.
"""

from .binding import CompiledScene, Record, compileScene, updateScene
from .scene import SceneError

__all__ = ["compileScene", "updateScene", "CompiledScene", "Record", "SceneError"]
