#pragma once

#include "types.h"
#include <cmath>


/*
Shapes do include
"Full shapes":
Elipse, Hyperbola, Parabola

"Fractioned shapes"
CircleArc, ElipseArc, HyperbolaArc, ParabolaArc, LineSegments.


*/

// LINE: a, b, c of a x + b y + c = 0, (a,b) kept unit

// LINE_SEGMENT: x0, y0, x1, y1, rest is calculated on the fly

// ELIPSE: x0, y0, a, b, theta, the elipse can be rotated

// ELIPSE_ARC: x0, y0, a, b, theta, phi0, phi1. phi0/1 are the beginning/end of the arc in radians, follwing the trigonometry configuration



enum ObjectType { LINE = 0, 
                  LINE_SEGMENT = 1,
                  ELIPSE = 2}; // Line for now, but will include Cicle, LineSegment, 
struct Object {
    int type;
    double p[4]; // LINE: a, b, c of a x + b y + c = 0, (a,b) kept unit
                 // LINE_SEGMENT: x0, y0, x1, y1
};

#ifdef _OPENMP
#pragma omp begin declare target
#endif

// ------------------------------------------------------------- 0. ballistics

// the state at time t, given the initial state s. Lives here rather than in
// physics.h because the segment needs the impact point to test its bounds
inline state trajectory(state s, double t) {
    return {s.x + s.vx * t,
            s.y + s.vy * t - 0.5 * g * t * t,
            s.vx,
            s.vy - g * t};
}

// ---------------------------------------------------------- 1. root finding

// smaller of two candidate times that is still ahead of us
inline double firstRoot(double t1, double t2) {
    double t = MAGIC_NO_COLLISION;
    if (t1 > deadTime && t1 < t) t = t1;
    if (t2 > deadTime && t2 < t) t = t2;
    return t;
}

// both roots of A t^2 + B t + C, MAGIC_NO_COLLISION where there is none.
// the pair is kept whole because a segment has to test each root separately
inline vector2D quadRoots(double A, double B, double C) {

    // special cases
    if (A == 0.0) {
        if (B == 0.0) return {MAGIC_NO_COLLISION, MAGIC_NO_COLLISION};
        return {-C / B, MAGIC_NO_COLLISION};
    }

    double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return {MAGIC_NO_COLLISION, MAGIC_NO_COLLISION};

    // stable form: compute q with the sign of B, so B + sq never cancels;
    // one root is q/A, the other C/q
    double sq = sqrt(disc);
    if (B < 0.0) {
        sq = -sq;
    }
    double q = -0.5 * (B + sq);
    if (q == 0.0) { // only when C is exactly 0: the roots are 0 and -B/A
        return {0.0, -B / A};
    }
    return {q / A, C / q};
}

// smallest root > deadTime of A t^2 + B t + C
inline double firstQuadRoot(double A, double B, double C) {
    vector2D r = quadRoots(A, B, C);
    return firstRoot(r.x, r.y);
}

// ------------------------------------------------------------------ 2. makers

// scale the coefficients so that (a, b) is the unit normal
inline Object makeLine(double a, double b, double c) {
    double len = sqrt(a * a + b * b);
    return {LINE, {a / len, b / len, c / len}};
}

// the endpoints are the whole story: the line through them is rebuilt on the fly
inline Object makeLineSegment(double x0, double y0, double x1, double y1) {
    return {LINE_SEGMENT, {x0, y0, x1, y1}};
}

// ------------------------------------------------------------------ 3. lines

// substitute the parabola into a x + b y + c: A t^2 + B t + C.
// a vertical line has b = 0, killing the t^2 term (linear branch)
inline vector2D lineRoots(double a, double b, double c, state s) {
    return quadRoots(-0.5 * g * b,
                     a * s.vx + b * s.vy,
                     a * s.x + b * s.y + c);
}

inline double lineHitTime(Object o, state s) {
    vector2D r = lineRoots(o.p[0], o.p[1], o.p[2], s);
    return firstRoot(r.x, r.y);
}

// same unit normal (a, b) everywhere (a circle's would depend on s)
inline vector2D lineNormal(Object o, state s) {
    return {o.p[0], o.p[1]};
}

// land exactly on the line: move by minus the signed distance, along the normal
inline state lineSnap(Object o, state s) {
    double d = o.p[0] * s.x + o.p[1] * s.y + o.p[2];
    return {s.x - d * o.p[0], s.y - d * o.p[1], s.vx, s.vy};
}

// --------------------------------------------------------------- 4. segments

// a segment is the line carrying it, plus a bound: everything below defers to
// the line, and only the hit time filters out what falls past the endpoints

// the infinite line through the two endpoints, (a, b) made unit by makeLine
inline Object segmentLine(Object o) {
    double x0 = o.p[0], y0 = o.p[1], x1 = o.p[2], y1 = o.p[3];
    return makeLine(y0 - y1, x1 - x0, x0 * y1 - x1 * y0);
}

// is the trajectory between the endpoints at time t? u runs along the segment,
// 0 at (x0, y0) and 1 at (x1, y1)
inline bool onSegment(Object o, double t, state s) {
    state at = trajectory(s, t);
    double dx = o.p[2] - o.p[0], dy = o.p[3] - o.p[1];
    double u = ((at.x - o.p[0]) * dx + (at.y - o.p[1]) * dy) / (dx * dx + dy * dy);
    return u >= 0.0 && u <= 1.0;
}

// the line's own roots, minus the ones that land off the ends. Both are needed:
// the earlier root can sail past an endpoint while the later one lands
inline double segmentHitTime(Object o, state s) {
    Object l = segmentLine(o);
    vector2D r = lineRoots(l.p[0], l.p[1], l.p[2], s);
    if (!onSegment(o, r.x, s)) r.x = MAGIC_NO_COLLISION;
    if (!onSegment(o, r.y, s)) r.y = MAGIC_NO_COLLISION;
    return firstRoot(r.x, r.y);
}

// no orientation to pick: bounce is quadratic in n, so a segment hits both ways
inline vector2D segmentNormal(Object o, state s) {
    return lineNormal(segmentLine(o), s);
}

inline state segmentSnap(Object o, state s) {
    return lineSnap(segmentLine(o), s);
}

// --------------------------------------------------------------- 5. dispatch

// time at which the trajectory meets the object, MAGIC_NO_COLLISION if never
inline double hitTime(Object o, state s) {
    switch (o.type) {
    case LINE: return lineHitTime(o, s);
    case LINE_SEGMENT: return segmentHitTime(o, s);
    }
    return MAGIC_NO_COLLISION;
}

// unit normal at the contact point
inline vector2D normalAt(Object o, state s) {
    switch (o.type) {
    case LINE: return lineNormal(o, s);
    case LINE_SEGMENT: return segmentNormal(o, s);
    }
    return {0.0, 0.0};
}

// project the leftover rounding back onto the surface
inline state snapTo(Object o, state s) {
    switch (o.type) {
    case LINE: return lineSnap(o, s);
    case LINE_SEGMENT: return segmentSnap(o, s);
    }
    return s;
}

#ifdef _OPENMP
#pragma omp end declare target
#endif
