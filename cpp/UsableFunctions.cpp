// the shared library's whole surface: everything ctypes calls, and everything
// here that has any logic in it. identical for every scene and every backend --
// only the compiler flags move -- so the scene arrives as a generated header
// holding nothing but a count, an array, and a list of maker calls.
//
//   g++    -O2 -ffp-contract=off        -shared -fPIC UsableFunctions.cpp
//   g++    -O2 -ffp-contract=off -fopenmp -foffload=disable -shared -fPIC ...
//   nvc++  -O2 -Kieee -Mnofma -mp=gpu -gpu=cc89 -shared -fPIC ...
#include "physics.h"
#include "generatedScene.h" // nObj, objs[], abBuildScene

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
    int     maxRows;   // stride of every column above
    int     columns;   // bitwise OR of abColumn
};

// which of the two tables below to fill. never crosses the ABI: each entry
// point picks its own, so a caller cannot ask for a mixture of the two
enum abMode { AB_EVENTS, AB_GRID };

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

// the two recorders run on the device, so they are declare target explicitly
// rather than leaning on the implicit rule for functions called from a target
// region: the GPU lane is the one nothing here would notice the loss of, since
// a build with no device kernel still runs perfectly well on the host
#ifdef _OPENMP
#pragma omp begin declare target
#endif

// one row of the caller's column arrays. the bitmask is read here and nowhere
// else, so both recorders write the same columns the same way
static inline void abWriteRow(const abOutput *o, int i, int n,
                              double t, state s, int kind) {
    const int cols = o->columns;
    long q = (long)i * o->maxRows + n;

    if (cols & AB_T)      o->t[q]      = t;
    if (cols & AB_X)      o->x[q]      = s.x;
    if (cols & AB_Y)      o->y[q]      = s.y;
    if (cols & AB_VX)     o->vx[q]     = s.vx;
    if (cols & AB_VY)     o->vy[q]     = s.vy;
    if (cols & AB_ENERGY) o->energy[q] = 0.5 * (s.vx * s.vx + s.vy * s.vy) + g * s.y;
    if (cols & AB_EVTYPE) o->evType[q] = kind;
}

// one particle, one row per event -- or per stride-th event, counting the
// initial state as event zero. stride decides only what is WRITTEN: every
// event in between is still simulated, so the trajectory is bit for bit the
// one a stride of 1 walks, just sampled coarsely.
//
// the caller sizes the buffer at nIter / stride + 1, which is exactly what
// this writes; n < maxRows is belt and braces against that arithmetic drifting
static int abRecordEvents(state s, int nIter, int stride, int i, const abOutput *o) {
    const int maxRows = o->maxRows;
    double t = 0.0;
    int n = 0, kind = EV_INITIAL;

    for (int k = 0, nextK = 0; k <= nIter && n < maxRows; ++k) {
        if (k == nextK) {
            abWriteRow(o, i, n, t, s, kind);
            ++n;
            nextK += stride; // additive: integer modulo is emulated on the GPU
        }
        if (k == nIter) break;

        double tc = stepScene(&s, objs, nObj, &kind);
        if (tc == MAGIC_NO_COLLISION) break; // nothing left to hit, ever
        t += tc;
    }
    return n;
}

// one particle, sampled on the uniform grid t = 0, dt, 2*dt, ... dt never
// enters the physics: the scene is still solved event to event, and dt only
// decides where we look. between two events the state comes from coasting the
// parabola forward from the one before, which is exact, so a small dt buys
// samples and nothing else -- no accuracy, because there was none to lose.
//
// maxEvents bounds the walk. a scene that traps a particle in unboundedly many
// collisions inside one dt would otherwise never reach the next sample, and a
// GPU kernel that does not finish is a great deal worse than an empty table:
// stopping early just leaves counts[i] short, which is how a run reports that
// it did not fill its rows
static int abSampleGrid(state s, double dt, int maxEvents, int i, const abOutput *o) {
    const int rows = o->maxRows;
    double t = 0.0;                // the time s is at: always an event
    int n = 0, kind = EV_INITIAL;  // how the flight we are inside of began

    for (int ev = 0; ev <= maxEvents && n < rows; ++ev) {
        // the step is taken before its cost is known, so keep what it started
        // from: every sample below belongs to the flight, not to where it ends
        state s0 = s;
        double t0 = t;

        int next = EV_BOUNCE;
        double tc = stepScene(&s, objs, nObj, &next);
        double tEnd = (tc == MAGIC_NO_COLLISION) ? MAGIC_NO_COLLISION : t0 + tc;

        // strictly before the event: a sample landing exactly on one is the
        // state just AFTER it, the same convention the event table uses.
        //
        // n * dt, never t += dt -- multiplying cannot drift over a million rows
        while (n < rows && n * dt < tEnd) {
            double tg = n * dt;
            abWriteRow(o, i, n, tg, trajectory(s0, tg - t0), kind);
            ++n;
        }

        if (tc == MAGIC_NO_COLLISION) break; // s never moved; the loop above
        t = tEnd;                            // already coasted out the rest
        kind = next;
    }
    return n;
}

#ifdef _OPENMP
#pragma omp end declare target
#endif

static int abRun(const double *x0, const double *y0,
                 const double *vx0, const double *vy0, int nPart,
                 int mode, int nIter, int stride, double dt, int maxEvents,
                 abOutput *o) {
    if (nPart < 1 || o->maxRows < 1) return 1;

    const int maxRows = o->maxRows, cols = o->columns;
    const long nCell = (long)nPart * maxRows;

    // hoisted so the map clauses name plain pointers, and sized to 1 for any
    // column the caller did not ask for
    double *ot = o->t, *ox = o->x, *oy = o->y;
    double *ovx = o->vx, *ovy = o->vy, *oe = o->energy;
    int *oev = o->evType, *cnt = o->counts;

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
               ovy[0 : lvy], oe[0 : le], oev[0 : lev], cnt[0 : nPart])
#endif
    for (int i = 0; i < nPart; ++i) {
        abOutput dev = {ot, ox, oy, ovx, ovy, oe, oev, cnt, maxRows, cols};
        state s = {x0[i], y0[i], vx0[i], vy0[i]};

        // one branch on a value every thread agrees on, rather than a second
        // copy of the twenty lines of map clauses above
        cnt[i] = (mode == AB_GRID) ? abSampleGrid(s, dt, maxEvents, i, &dev)
                                   : abRecordEvents(s, nIter, stride, i, &dev);
    }
    return 0;
}

// two entry points rather than one with a mode flag: they do not record the
// same table. one row per event is not a coarse version of one row per tick,
// and no argument list that offered both could say which you meant
extern "C" int abRunIterations(const double *x0, const double *y0,
                               const double *vx0, const double *vy0,
                               int nPart, int nIter, int stride, abOutput *o) {
    if (nIter < 0 || stride < 1) return 1;
    return abRun(x0, y0, vx0, vy0, nPart, AB_EVENTS, nIter, stride, 0.0, 0, o);
}

extern "C" int abRunSampled(const double *x0, const double *y0,
                            const double *vx0, const double *vy0,
                            int nPart, double dt, int maxEvents, abOutput *o) {
    // !(dt > 0) also catches NaN, which would leave the sampler's inner loop
    // spinning with no sample ever landing before the next event
    if (!(dt > 0.0) || maxEvents < 0) return 1;
    return abRun(x0, y0, vx0, vy0, nPart, AB_GRID, 0, 1, dt, maxEvents, o);
}
