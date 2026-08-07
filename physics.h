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

inline state teleport(Object o, state s) {
    Frame a = frameFromPoints(o.q[0], o.q[1], o.q[2], o.q[3]); // this portal
    Frame b = frameFromPoints(o.q[4], o.q[5], o.q[6], o.q[7]); // the far side
    double flip = o.q[8];

    // the same distance along the exit: the pair is the same size, so an end
    // still comes out at an end
    double u = (s.x - a.o.x) * a.t.x + (s.y - a.o.y) * a.t.y;

    double vt = s.vx * a.t.x + s.vy * a.t.y;
    double vn = s.vx * a.n.x + s.vy * a.n.y;

    return {b.o.x + u * b.t.x,
            b.o.y + u * b.t.y,
            vt * b.t.x + flip * vn * b.n.x,
            vt * b.t.y + flip * vn * b.n.y};
}

inline state respond(Object o, state s) {
    switch (o.response) {
    case REFLECT: return bounce(o, s);
    case PORTAL: return teleport(o, s);
    }
    return s;
}


// cap is how many rows out can hold. a portal event needs two of them, so size
// it for 2 * maxB + 1 to be sure of getting the full run
inline int getCollisionStates(state s, const Object *objs, int nObj, int maxB,
                              Row *out, int cap) {
    double t = 0.0;
    int nRowsWritten = 0;

    out[nRowsWritten++] = {t, s.x, s.y, s.vx, s.vy}; // the initial conditions

    for (int i = 0; i < maxB; ++i) {

        // the earliest hit wins. objects that are never met answer with the
        // sentinel, which dwarfs any real time and so loses every comparison
        double tc = MAGIC_NO_COLLISION;
        for (int k = 0; k < nObj; ++k) {
            double tk = hitTime(objs[k], s);
            if (tk < tc) tc = tk;
        }
        if (tc == MAGIC_NO_COLLISION) break;

        state at = trajectory(s, tc);
        t += tc;

        // a portal moves the particle elsewhere, so nothing else touched back at
        // the old spot still applies: the first one found is the whole event
        int portal = -1;
        for (int k = 0; k < nObj; ++k) {
            if (objs[k].response == PORTAL && hitTime(objs[k], s) <= tc + deadTime) {
                portal = k;
                break;
            }
        }

        // room is checked per branch: reserving two rows for every event would cut
        // a portal-free run one event short of what was asked for
        if (portal >= 0) {
            if (nRowsWritten + 2 > cap) break;
            state arrival = snapTo(objs[portal], at);
            // recorded in its own right, or the flight in would appear to end
            // wherever the particle came out, and consecutive rows would no
            // longer be one ballistic arc apart
            out[nRowsWritten++] = {t, arrival.x, arrival.y, arrival.vx, arrival.vy};
            s = teleport(objs[portal], arrival);
        } else {
            if (nRowsWritten + 1 > cap) break;
            // every object within deadTime of tc, not just the first: a corner is
            // one impact against both walls, and resolving only one would leave
            // the particle on the other moving outward
            for (int k = 0; k < nObj; ++k) {
                if (hitTime(objs[k], s) <= tc + deadTime) {
                    at = bounce(objs[k], snapTo(objs[k], at));
                }
            }
            s = at;
        }
        out[nRowsWritten++] = {t, s.x, s.y, s.vx, s.vy};
    }
    return nRowsWritten;
}

// event to event up to t_f, then coast the leftover time. keeps only the final
// state, so it costs O(1) memory however long the run
inline int getFinalPosition(state s, const Object *objs, int nObj, double t_f, state *out) {
    double t = 0.0;
    int nBounces = 0;

    for (;;) {
        double tc = MAGIC_NO_COLLISION;
        for (int k = 0; k < nObj; ++k) {
            double tk = hitTime(objs[k], s);
            if (tk < tc) tc = tk;
        }

        // next collision past t_f, or never (the sentinel dwarfs any t_f):
        // coast the remaining time and we are done
        if (t + tc > t_f) {
            *out = trajectory(s, t_f - t);
            return nBounces;
        }

        state at = trajectory(s, tc);

        // resolved exactly as in getCollisionStates: a portal is exclusive,
        // anything else takes every object within deadTime of tc
        int portal = -1;
        for (int k = 0; k < nObj; ++k) {
            if (objs[k].response == PORTAL && hitTime(objs[k], s) <= tc + deadTime) {
                portal = k;
                break;
            }
        }

        if (portal >= 0) {
            s = teleport(objs[portal], snapTo(objs[portal], at));
        } else {
            for (int k = 0; k < nObj; ++k) {
                if (hitTime(objs[k], s) <= tc + deadTime) {
                    at = bounce(objs[k], snapTo(objs[k], at));
                }
            }
            s = at;
        }
        t += tc;
        ++nBounces;
    }
}






#ifdef _OPENMP
#pragma omp end declare target
#endif
