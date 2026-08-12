// the shared library's whole surface: everything ctypes calls, and everything
// here that has any logic in it. identical for every scene and every backend --
// only the compiler flags move -- so the scene arrives as a generated header
// holding nothing but a count, an array, and a list of maker calls.
//
//   g++    -O2 -ffp-contract=off        -shared -fPIC UsableFunctions.cpp
//   g++    -O2 -ffp-contract=off -fopenmp -foffload=disable -shared -fPIC ...
//   nvc++  -O2 -Kieee -Mnofma -mp=gpu -gpu=cc120 -shared -fPIC ...
#include "physics.h"
#include "generatedScene.h" // nObj, objs[], abBuildScene
#include <climits>

// which columns the caller wants. every pointer below must be valid even when
// unwanted, because OpenMP cannot map a null one -- the binding passes a
// one-element throwaway for those and leaves the bit clear, so nothing is
// written to it and nothing large is allocated for it
enum abColumn {
    AB_T      = 1 << 0, AB_X      = 1 << 1, AB_Y  = 1 << 2,
    AB_VX     = 1 << 3, AB_VY     = 1 << 4,
    AB_ENERGY = 1 << 5, AB_EVTYPE = 1 << 6
};

struct abOutput {
    double *t, *x, *y, *vx, *vy, *energy;
    int    *evType;
    int    *counts;    // rows actually written, per particle
    int    *truncated; // per particle: did it stop because maxRows ran out
    int     maxRows;   // stride of every column above
    int     columns;   // bitwise OR of abColumn
};

extern "C" int abObjectCount() { return nObj; }

// g and deadTime live in types.h as mutable globals so they can be set without
// a rebuild. deadTime < 0 would let firstRoot re-find the collision just
// resolved, and the solver would sit on it forever
extern "C" int abSetConstants(double gravity, double dead) {
    if (!(dead >= 0.0)) return 1; // also catches NaN
    g = gravity;
    deadTime = dead;
#ifdef _OPENMP
#pragma omp target update to(g, deadTime)
#endif
    return 0;
}

// one particle, recorded into the caller's column arrays. capped by event count
// and by time at once: whichever entry point is unused passes the identity for
// its own cap, so there is one loop rather than two nearly-identical ones
static int abRecordOne(state s, int maxB, double tf, int i, const abOutput *o) {
    const int maxRows = o->maxRows, cols = o->columns;
    double t = 0.0;
    int n = 0, kind = EV_INITIAL;

    for (int k = 0; k <= maxB; ++k) {
        if (n >= maxRows) return -n; // negative marks truncation

        long q = (long)i * maxRows + n;
        if (cols & AB_T)      o->t[q]      = t;
        if (cols & AB_X)      o->x[q]      = s.x;
        if (cols & AB_Y)      o->y[q]      = s.y;
        if (cols & AB_VX)     o->vx[q]     = s.vx;
        if (cols & AB_VY)     o->vy[q]     = s.vy;
        if (cols & AB_ENERGY) o->energy[q] = 0.5 * (s.vx * s.vx + s.vy * s.vy) + g * s.y;
        if (cols & AB_EVTYPE) o->evType[q] = kind;
        ++n;

        if (k == maxB) break;
        double tc = stepScene(&s, objs, nObj, &kind);
        if (tc == MAGIC_NO_COLLISION || t + tc > tf) break;
        t += tc;
    }
    return n;
}

static int abRun(const double *x0, const double *y0,
                 const double *vx0, const double *vy0,
                 int nPart, int maxB, double tf, abOutput *o) {
    if (nPart < 1 || o->maxRows < 1) return 1;

    const int maxRows = o->maxRows, cols = o->columns;
    const long nCell = (long)nPart * maxRows;

    // hoisted so the map clauses name plain pointers, and sized to 1 for any
    // column the caller did not ask for
    double *ot = o->t, *ox = o->x, *oy = o->y;
    double *ovx = o->vx, *ovy = o->vy, *oe = o->energy;
    int *oev = o->evType, *cnt = o->counts, *trn = o->truncated;

    const long lt  = (cols & AB_T)      ? nCell : 1;
    const long lx  = (cols & AB_X)      ? nCell : 1;
    const long ly  = (cols & AB_Y)      ? nCell : 1;
    const long lvx = (cols & AB_VX)     ? nCell : 1;
    const long lvy = (cols & AB_VY)     ? nCell : 1;
    const long le  = (cols & AB_ENERGY) ? nCell : 1;
    const long lev = (cols & AB_EVTYPE) ? nCell : 1;

    // objs, g and deadTime are declare target globals refreshed by
    // abBuildScene / abSetConstants, so they are not mapped here
#ifdef _OPENMP
#pragma omp target teams distribute parallel for                            \
    map(to : x0[0 : nPart], y0[0 : nPart], vx0[0 : nPart], vy0[0 : nPart])  \
    map(from : ot[0 : lt], ox[0 : lx], oy[0 : ly], ovx[0 : lvx],            \
               ovy[0 : lvy], oe[0 : le], oev[0 : lev],                      \
               cnt[0 : nPart], trn[0 : nPart])
#endif
    for (int i = 0; i < nPart; ++i) {
        abOutput dev = {ot, ox, oy, ovx, ovy, oe, oev, cnt, trn, maxRows, cols};
        int n = abRecordOne({x0[i], y0[i], vx0[i], vy0[i]}, maxB, tf, i, &dev);
        trn[i] = n < 0;
        cnt[i] = n < 0 ? -n : n;
    }
    return 0;
}

// two entry points rather than one with a mode flag, so that "iterations or tf,
// never both" cannot even be expressed here
extern "C" int abRunIterations(const double *x0, const double *y0,
                               const double *vx0, const double *vy0,
                               int nPart, int nIter, abOutput *o) {
    if (nIter < 0) return 1;
    return abRun(x0, y0, vx0, vy0, nPart, nIter, MAGIC_NO_COLLISION, o);
}

extern "C" int abRunTime(const double *x0, const double *y0,
                         const double *vx0, const double *vy0,
                         int nPart, double tf, abOutput *o) {
    if (!(tf >= 0.0)) return 1;
    return abRun(x0, y0, vx0, vy0, nPart, INT_MAX, tf, o);
}
