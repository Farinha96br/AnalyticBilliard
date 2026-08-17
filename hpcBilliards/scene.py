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

# a basin shape is the same geometry with a different response, so it reuses the
# maker -- but "a line you bounce off" and "a line that ends the run" are not the
# same structure, so the prefix keeps them apart in the hash. the label rides
# along as one more parameter, which is what lets basins be relabelled, and
# moved, without a rebuild
BASIN = "basin:"
for _k, _n in list(NPARAM.items()):
    if _k in MAKER:
        NPARAM[BASIN + _k] = _n + 1


class SceneError(ValueError):
    pass


def _basinSlots(scene):
    """basinObjects, a dict of label -> shapes, flattened in sorted label order.

    Sorted so that the same dict always produces the same structure hash: dict
    order is insertion order, and two callers building the same basins in a
    different order must not compile two libraries.
    """
    basins = scene.get("basinObjects") or {}
    if not isinstance(basins, dict):
        raise SceneError(f"basinObjects must be a dict of label -> list of "
                         f"shapes, got {type(basins).__name__}")

    out = []
    for label in sorted(basins):
        if not isinstance(label, int) or isinstance(label, bool):
            raise SceneError(f"basin label {label!r} must be an int")
        if label < 1:
            raise SceneError(f"basin label {label} is not usable: 0 is reserved "
                             f"for a particle that reached tf without escaping, "
                             f"so labels start at 1")
        shapes = basins[label]
        if not shapes:
            raise SceneError(f"basin {label} has no shapes")
        for i, o in enumerate(shapes):
            kind = o.get("type")
            if kind not in MAKER:
                raise SceneError(f"basinObjects[{label}][{i}]: unknown type "
                                 f"{kind!r}, expected one of {sorted(MAKER)}")
            # checked here, not in flatten: there the label is already appended
            # and the count it reports would be one more than what was written
            params = list(o.get("params", []))
            if len(params) != NPARAM[kind]:
                raise SceneError(f"basinObjects[{label}][{i}] ({kind}): expected "
                                 f"{NPARAM[kind]} parameters, got {len(params)}")
            out.append((BASIN + kind, params + [float(label)]))
    return out


def _slots(scene):
    """the scene as a flat list of (type, params): solids, portals, then basins"""
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

    out.extend(_basinSlots(scene))
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
