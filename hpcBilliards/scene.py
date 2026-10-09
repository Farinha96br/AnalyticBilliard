
import hashlib
from dataclasses import dataclass, field, fields


NPARAM = {}  # kind -> parameter count, filled in from the shapes below

MAKER = {
    "line":        "makeLine",
    "lineSegment": "makeLineSegment",
    "elipse":      "makeElipse",
    "elipseArc":   "makeElipseArc",
}

# same geometry, different response. kept apart in the hash; label is a param
BASIN = "basin:"


class SceneError(ValueError):
    pass


# a shape's fields ARE its params, in the order codegen reads them out of p[].
# validate() mirrors the C guards in codegen.py, but earlier and by name.


@dataclass
class Line:
    """a*x + b*y + c = 0, infinite"""
    a: float
    b: float
    c: float

    kind = "line"

    def params(self):
        return [self.a, self.b, self.c]

    def __post_init__(self):
        self.validate()

    def validate(self, where=None):
        # the sum must be non-zero, not a and b: it underflows while both are
        # still finite, which the C guard (a == 0 && b == 0) misses
        if self.a * self.a + self.b * self.b == 0.0:
            raise SceneError(f"{where or 'Line'}: a*a + b*b underflows to zero "
                             f"(a={self.a}, b={self.b}), so the normal would be "
                             f"inf and nothing would ever hit it")


@dataclass
class LineSegment:
    """(x0, y0) to (x1, y1)"""
    x0: float
    y0: float
    x1: float
    y1: float

    kind = "lineSegment"

    def params(self):
        return [self.x0, self.y0, self.x1, self.y1]

    def __post_init__(self):
        self.validate()

    def validate(self, where=None):
        if self.x0 == self.x1 and self.y0 == self.y1:
            raise SceneError(f"{where or 'LineSegment'}: zero length, both ends "
                             f"at ({self.x0}, {self.y0})")


@dataclass
class Elipse:
    """centre (cx, cy), radii a and b, rotated by tilt radians"""
    cx: float
    cy: float
    a: float
    b: float
    tilt: float

    kind = "elipse"

    def params(self):
        return [self.cx, self.cy, self.a, self.b, self.tilt]

    def __post_init__(self):
        self.validate()

    def validate(self, where=None):
        if self.a == 0.0 or self.b == 0.0:
            raise SceneError(f"{where or type(self).__name__}: zero radius "
                             f"(a={self.a}, b={self.b})")


@dataclass
class ElipseArc(Elipse):
    """an Elipse from phi0 to phi1, radians. phi1 under phi0 wraps through zero"""
    phi0: float
    phi1: float

    kind = "elipseArc"

    def params(self):
        return super().params() + [self.phi0, self.phi1]


@dataclass
class Portal:
    """enter by `entry`, leave by `exit`. the exit is a destination, not a wall"""
    entry: LineSegment
    exit: LineSegment
    normalFlip: float = 1.0
    tangentFlip: float = 1.0

    kind = "portal"

    def params(self):
        return (self.entry.params() + self.exit.params()
                + [self.normalFlip, self.tangentFlip])

    def __post_init__(self):
        self.validate()

    def validate(self, where=None):
        where = where or "Portal"
        for name in ("entry", "exit"):
            side = getattr(self, name)
            if not isinstance(side, LineSegment):
                raise SceneError(f"{where}: Portal.{name} must be a "
                                 f"LineSegment, got {type(side).__name__}")
            side.validate(f"{where}.{name}")


SHAPES = (Line, LineSegment, Elipse, ElipseArc)

# from the classes, so a new field cannot leave codegen on the old count
NPARAM.update({C.kind: len(fields(C)) for C in SHAPES})
NPARAM["portal"] = 10  # the exception: 4 fields, two of them segments
for _k, _n in list(NPARAM.items()):
    if _k in MAKER:
        NPARAM[BASIN + _k] = _n + 1


@dataclass(kw_only=True)
class Scene:
    """What is in the billiard, and under what gravity.

    Keyword-only: the five fields are not interchangeable. Mutable, so a sweep
    edits and pushes. CompiledScene keeps the scene it was given, so an edit
    after compiling reaches the library only once updateScene runs.
    """
    solidObjects: list = field(default_factory=list)
    portalObjects: list = field(default_factory=list)
    basinObjects: dict = field(default_factory=dict)
    g: float = 1.0
    deadTime: float = 1e-9

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Raise SceneError unless every object and constant is usable.

        Runs at construction and at each door into the library, since the
        objects are mutable. Idempotent and O(objects).
        """
        for i, o in enumerate(self.solidObjects):
            if not isinstance(o, SHAPES):
                raise SceneError(
                    f"solidObjects[{i}]: expected one of "
                    f"{[C.__name__ for C in SHAPES]}, got {type(o).__name__}")
            o.validate(f"solidObjects[{i}]")

        for i, p in enumerate(self.portalObjects):
            if not isinstance(p, Portal):
                raise SceneError(f"portalObjects[{i}]: expected a Portal, got "
                                 f"{type(p).__name__}")
            p.validate(f"portalObjects[{i}]")

        if not isinstance(self.basinObjects, dict):
            raise SceneError(f"basinObjects must be a dict of label -> list of "
                             f"shapes, got {type(self.basinObjects).__name__}")
        for label, shapes in self.basinObjects.items():
            if not isinstance(label, int) or isinstance(label, bool):
                raise SceneError(f"basin label {label!r} must be an int")
            if label < 1:
                raise SceneError(f"basin label {label} is not usable: 0 is "
                                 f"reserved for a particle that reached tf "
                                 f"without escaping, so labels start at 1")
            if not shapes:
                raise SceneError(f"basin {label} has no shapes")
            for i, o in enumerate(shapes):
                if not isinstance(o, SHAPES):
                    raise SceneError(
                        f"basinObjects[{label}][{i}]: expected one of "
                        f"{[C.__name__ for C in SHAPES]}, got "
                        f"{type(o).__name__}")
                o.validate(f"basinObjects[{label}][{i}]")

        if not float(self.deadTime) >= 0.0:
            raise SceneError(f"deadTime must be >= 0, got {self.deadTime}: a "
                             f"negative one lets the solver re-find the "
                             f"collision it just resolved, and the run hangs")

    def slots(self):
        """-> [(kind, params)]: solids, portals, then basins in label order"""
        out = [(o.kind, o.params()) for o in self.solidObjects]
        out += [(p.kind, p.params()) for p in self.portalObjects]
        # sorted, so the same basins built in a different order hash the same
        for label in sorted(self.basinObjects):
            for o in self.basinObjects[label]:
                out.append((BASIN + o.kind, o.params() + [float(label)]))
        return out


def _asScene(scene, who):
    if isinstance(scene, Scene):
        return scene
    if isinstance(scene, dict):
        raise SceneError(
            f"{who} takes a Scene, not a dict. A scene is now built from typed "
            f"objects, so a missing angle or a misspelled key is an error where "
            f"you write it:\n"
            f"    Scene(g=0.5, solidObjects=[LineSegment(0, 0, 1, 0)])\n"
            f"Import them from hpcBilliards.")
    raise SceneError(f"{who} takes a Scene, got {type(scene).__name__}")


def flatten(scene):
    """-> (types, params, offsets). params is every object's run end to end."""
    slots = scene.slots()
    if not slots:
        raise SceneError("scene has no objects")

    types, params, offsets = [], [], []
    for kind, p in slots:
        offsets.append(len(params))
        params.extend(float(v) for v in p)
        types.append(kind)
    return types, params, offsets


def structure_hash(scene):
    """the slot types in order, nothing else: a parameter must not rebuild"""
    types = [k for k, _ in scene.slots()]
    return hashlib.sha1("|".join(types).encode()).hexdigest(), types


def constants(scene):
    return float(scene.g), float(scene.deadTime)
