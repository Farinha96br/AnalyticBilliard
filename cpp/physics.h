#pragma once

#include "geometry.h" // pulls in types.h

#ifdef _OPENMP
#pragma omp begin declare target
#endif

inline state bounce(Object o, state s) {
    vector2D n = normalAt(o, s);
    double vn = s.vx * n.x + s.vy * n.y;
    return {s.x, s.y,
            s.vx - (1.0 + e) * vn * n.x,
            s.vy - (1.0 + e) * vn * n.y};
}

// no length is kept: asPortal builds the pair at matching sizes, so the map is a
// rigid motion and distance along the surface is all that crosses over
struct Frame {
    vector2D o, t, n;
};

inline Frame frameFromPoints(double x0, double y0, double x1, double y1) {
    double dx = x1 - x0, dy = y1 - y0;
    double len = sqrt(dx * dx + dy * dy);
    vector2D t = {dx / len, dy / len};
    return {{x0, y0}, t, {-t.y, t.x}};
}

// the two signs act on the VELOCITY and nothing else; the position comes from
// the frames alone. that is what makes all four sign pairs land on the same
// point, and so conserve energy on a level pair
inline state teleport(Object o, state s) {
    Frame a = frameFromPoints(o.q[0], o.q[1], o.q[2], o.q[3]); // this portal
    Frame b = frameFromPoints(o.q[4], o.q[5], o.q[6], o.q[7]); // the far side
    double flipNormal = o.q[8], flipTangent = o.q[9];

    double u = (s.x - a.o.x) * a.t.x + (s.y - a.o.y) * a.t.y;

    double vt = s.vx * a.t.x + s.vy * a.t.y;
    double vn = s.vx * a.n.x + s.vy * a.n.y;

    return {b.o.x + u * b.t.x,
            b.o.y + u * b.t.y,
            flipTangent * vt * b.t.x + flipNormal * vn * b.n.x,
            flipTangent * vt * b.t.y + flipNormal * vn * b.n.y};
}

inline state respond(Object o, state s) {
    switch (o.response) {
    case REFLECT: return bounce(o, s);
    case PORTAL: return teleport(o, s);
    }
    return s;
}


enum EventType { EV_INITIAL = 0, EV_BOUNCE = 1, EV_PORTAL = 2 };

// advance s to the next event. returns the time it took, or MAGIC_NO_COLLISION
// if nothing is ever hit; *kind says which sort of event it was.
//
// both recorders share this so the two subtleties below are written once. they
// were each a bug at some point and neither is obvious from the outside
inline double stepScene(state *s, const Object *objs, int nObj, int *kind) {
    // objects never met answer with the sentinel, which loses every comparison
    double tc = MAGIC_NO_COLLISION;
    for (int k = 0; k < nObj; ++k) {
        double tk = hitTime(objs[k], *s);
        if (tk < tc) tc = tk;
    }
    if (tc == MAGIC_NO_COLLISION) return tc;

    state at = trajectory(*s, tc);

    int portal = -1;
    for (int k = 0; k < nObj; ++k) {
        if (objs[k].response == PORTAL && hitTime(objs[k], *s) <= tc + deadTime) {
            portal = k;
            break;
        }
    }

    // every ordinary surface within deadTime of tc, not just the first: a
    // corner is one impact against both walls.
    //
    // this runs even when a portal fired. a portal moves the particle, so it
    // has to go LAST, but a wall genuinely touched here still turns the
    // velocity -- skip it and a portal ending on a wall carries the particle
    // away still travelling into the surface, and out of the scene
    for (int k = 0; k < nObj; ++k) {
        if (objs[k].response != PORTAL && hitTime(objs[k], *s) <= tc + deadTime) {
            at = bounce(objs[k], snapTo(objs[k], at));
        }
    }

    if (portal >= 0) {
        *s = teleport(objs[portal], snapTo(objs[portal], at));
        *kind = EV_PORTAL;
    } else {
        *s = at;
        *kind = EV_BOUNCE;
    }
    return tc;
}

// one row per event, so cap = maxB + 1 holds a full run.
//
// a portal writes only where the particle came OUT. where it went in is not
// stored because integrating from the previous row for t[i+1] - t[i] lands on
// it exactly -- storing it too would be the one redundancy this table avoids
inline int getCollisionStates(state s, const Object *objs, int nObj, int maxB,
                              Row *out, int cap, int *evType = nullptr) {
    double t = 0.0;
    int n = 0;

    out[n] = {t, s.x, s.y, s.vx, s.vy};
    if (evType) evType[n] = EV_INITIAL;
    ++n;

    for (int i = 0; i < maxB && n < cap; ++i) {
        int kind = EV_BOUNCE;
        double tc = stepScene(&s, objs, nObj, &kind);
        if (tc == MAGIC_NO_COLLISION) break;

        t += tc;
        out[n] = {t, s.x, s.y, s.vx, s.vy};
        if (evType) evType[n] = kind;
        ++n;
    }
    return n;
}

// the same recording, capped by time rather than by event count. the step is
// taken before its cost is known, so an overshoot is simply not recorded
inline int getCollisionStatesUntil(state s, const Object *objs, int nObj, double t_f,
                                   Row *out, int cap, int *evType = nullptr) {
    double t = 0.0;
    int n = 0;

    out[n] = {t, s.x, s.y, s.vx, s.vy};
    if (evType) evType[n] = EV_INITIAL;
    ++n;

    while (n < cap) {
        int kind = EV_BOUNCE;
        double tc = stepScene(&s, objs, nObj, &kind);
        if (tc == MAGIC_NO_COLLISION || t + tc > t_f) break;

        t += tc;
        out[n] = {t, s.x, s.y, s.vx, s.vy};
        if (evType) evType[n] = kind;
        ++n;
    }
    return n;
}

// keeps only the final state, so it costs O(1) memory however long the run
inline int getFinalPosition(state s, const Object *objs, int nObj, double t_f, state *out) {
    double t = 0.0;
    int nBounces = 0;

    for (;;) {
        double tc = MAGIC_NO_COLLISION;
        for (int k = 0; k < nObj; ++k) {
            double tk = hitTime(objs[k], s);
            if (tk < tc) tc = tk;
        }

        // past t_f, or never (the sentinel dwarfs any t_f): coast and finish
        if (t + tc > t_f) {
            *out = trajectory(s, t_f - t);
            return nBounces;
        }

        state at = trajectory(s, tc);

        // resolved exactly as in getCollisionStates
        int portal = -1;
        for (int k = 0; k < nObj; ++k) {
            if (objs[k].response == PORTAL && hitTime(objs[k], s) <= tc + deadTime) {
                portal = k;
                break;
            }
        }
        for (int k = 0; k < nObj; ++k) {
            if (objs[k].response != PORTAL && hitTime(objs[k], s) <= tc + deadTime) {
                at = bounce(objs[k], snapTo(objs[k], at));
            }
        }
        s = (portal >= 0) ? teleport(objs[portal], snapTo(objs[portal], at)) : at;
        t += tc;
        ++nBounces;
    }
}

#ifdef _OPENMP
#pragma omp end declare target
#endif
