"""AnalyticBilliard as a shared library.

    from hpcBilliards import compileScene, updateScene

    compiled = compileScene(SCENE, backend="openmp")
    rec = compiled.recordScene(x, y, vx, vy, iterations=1000)

Declare the objects once and compile; after that, parameters, the portal flips,
gravity and deadTime all change without touching the compiler. Only adding,
removing or retyping an object needs a rebuild.
"""

from .binding import CompiledScene, Record, compileScene, updateScene
from .scene import SceneError

__all__ = ["compileScene", "updateScene", "CompiledScene", "Record", "SceneError"]
