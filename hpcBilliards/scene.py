"""Validate a scene dict, flatten its parameters, and hash its structure.

The split between structure and parameters is the whole design: structure is
baked into generated C++ and needs a rebuild to change, parameters are pushed
into the running library. So only structure is hashed.
"""

import hashlib

# params per object type. a portal carries its entry (4), its exit (4) and the
# two flips -- the exit is a destination, not a surface, so it is not an object
# of its own and nothing bounces off it
NPARAM = {"line": 3, "lineSegment": 4, "elipse": 5, "elipseArc": 7, "portal": 10}

MAKER = {
    "line":        "makeLine",
    "lineSegment": "makeLineSegment",
    "elipse":      "makeElipse",
    "elipseArc":   "makeElipseArc",
}


class SceneError(ValueError):
    pass


def _slots(scene):
    """the scene as a flat list of (type, params), solids first then portals"""
    out = []
    for i, o in enumerate(scene.get("solidObjects", [])):
        kind = o.get("type")
        if kind not in MAKER:
            raise SceneError(f"solidObjects[{i}]: unknown type {kind!r}, "
                             f"expected one of {sorted(MAKER)}")
        out.append((kind, list(o.get("params", []))))

    for i, p in enumerate(scene.get("portalObjects", [])):
        for key in ("entry", "exit"):
            if p.get(key) != "lineSegment":
                raise SceneError(f"portalObjects[{i}]: {key!r} must be "
                                 f"'lineSegment', got {p.get(key)!r}")
        out.append(("portal", list(p["entryParams"]) + list(p["exitParams"])
                    + [float(p.get("normalFlip", 1.0)),
                       float(p.get("tangentFlip", 1.0))]))
    return out


def flatten(scene):
    """-> (types, params, offsets). params is every object's run end to end."""
    slots = _slots(scene)
    if not slots:
        raise SceneError("scene has no objects")

    types, params, offsets = [], [], []
    for k, (kind, p) in enumerate(slots):
        want = NPARAM[kind]
        if len(p) != want:
            raise SceneError(f"object {k} ({kind}): expected {want} parameters, "
                             f"got {len(p)}")
        offsets.append(len(params))
        params.extend(float(v) for v in p)
        types.append(kind)
    return types, params, offsets


def structure_hash(scene):
    """what a rebuild depends on: the slot types, in order, and nothing else.

    deliberately excludes every parameter. hashing those too would make tilting
    a line look like a new scene and trigger the rebuild this exists to avoid.
    """
    types = [k for k, _ in _slots(scene)]
    return hashlib.sha1("|".join(types).encode()).hexdigest(), types


def constants(scene):
    g = float(scene.get("g", 1.0))
    dead = float(scene.get("deadTime", 1e-9))
    if not dead >= 0.0:
        raise SceneError(f"deadTime must be >= 0, got {dead}: a negative one "
                         f"lets the solver re-find the collision it just "
                         f"resolved, and the run hangs")
    return g, dead
