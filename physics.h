#pragma once
/*
        1. reflection    mirror the velocity about the surface normal
        2. simulation    the per-particle loop (CPU loop == GPU kernel)

        the ballistics (trajectory) live in geometry.h, where the shapes
        that have to sample the parabola can reach them
*/


#include "geometry.h" // pulls in types.h

#ifdef _OPENMP
#pragma omp begin declare target
#endif


// 1.

// 1.1 Reflection
// mirror the velocity about the normal

inline state bounce(Object o, state s) {
    vector2D n = normalAt(o, s);
    double vn = s.vx * n.x + s.vy * n.y;
    return {s.x, s.y,
            s.vx - (1.0 + e) * vn * n.x,
            s.vy - (1.0 + e) * vn * n.y};
}

//2. simulation
// Simulate the bounces of a particle: bounce off whichever object comes first






inline int getCollisionStates(state s, const Object *objs, int nObj, int maxB, Row *out) {
    double t = 0.0; // initial time
    int nRowsWritten = 0; // number of rows written to the output buffer

    // include the initial conditions
    out[nRowsWritten++] = {t, s.x, s.y, s.vx, s.vy}; 


    // actual loop for integration
    for (int i = 0; i < maxB; ++i) {

        // check collision time with all objects
        // use the MAGIC_NO_COLLISION to filter the objects that never hit the magic time (I hope)
        
        double tc = MAGIC_NO_COLLISION;
        for (int k = 0; k < nObj; ++k) {
            double tk = hitTime(objs[k], s);
            if (tk < tc) tc = tk;
        }
        if (tc == MAGIC_NO_COLLISION) break; 

        // move to the time of the collision event
        state at = trajectory(s, tc);
        for (int k = 0; k < nObj; ++k) {
            if (hitTime(objs[k], s) <= tc + deadTime) {
                at = bounce(objs[k], snapTo(objs[k], at));
            }
        }
        s = at;
        t += tc;
        out[nRowsWritten++] = {t, s.x, s.y, s.vx, s.vy};
    }
    return nRowsWritten;
}

// evolve a particle up to the final time t_f: bounce collision to collision
// while they happen before t_f, then coast the leftover time. *out is the
inline int getFinalPosition(state s, const Object *objs, int nObj, double t_f, state *out) {
    double t = 0.0; // initial time
    int nBounces = 0;


    // endless loop with cursed notation
    for (;;) {

        // check collision time with all objects
        // use the MAGIC_NO_COLLISION to filter the objects that never hit algebraically (hopefully)

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

        // move to the time of the collision event
        // deadTime of it: a corner is one impact against both walls, and resolving
        // only one would leave the particle on the other wall moving outward,
        // with the escaping root then filtered by deadTime as "already resolved"
        state at = trajectory(s, tc);
        for (int k = 0; k < nObj; ++k) {
            if (hitTime(objs[k], s) <= tc + deadTime) {
                at = bounce(objs[k], snapTo(objs[k], at));
            }
        }
        s = at;
        t += tc;
        ++nBounces;
    }
}






#ifdef _OPENMP
#pragma omp end declare target
#endif
