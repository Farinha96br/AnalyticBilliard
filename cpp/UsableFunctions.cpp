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

#ifdef _OPENMP
#include <omp.h> // omp_is_initial_device, for abOnDevice below
#endif

// which columns the caller wants. every pointer below must be valid even when
// unwanted, because OpenMP cannot map a null one -- the binding passes a
// one-element throwaway for those and leaves the bit clear, so nothing is
// written to it and nothing large is allocated for it
enum abColumn {
    AB_T      = 1 << 0, AB_X      = 1 << 1, AB_Y  = 1 << 2,
    AB_VX     = 1 << 3, AB_VY     = 1 << 4,
    AB_ENERGY = 1 << 5, AB_EVTYPE = 1 << 6, AB_BASIN = 1 << 7
};

struct abOutput {
    double *t, *x, *y, *vx, *vy, *energy;
    int    *evType;
    int    *counts;    // rows actually written, per particle
    int     maxRows;   // stride of every column above
    int     columns;   // bitwise OR of abColumn
};

// where a run ended, one row per particle rather than a table: an escape run
// keeps no trajectory at all, so its cost in memory is the same whether it ran
// for one collision or a million
struct abEscape {
    double *t, *x, *y, *vx, *vy, *energy;
    int    *basin;   // which basin absorbed it, or 0 for "still going at tf"
    int     columns; // bitwise OR of abColumn
};

// which of the two tables below to fill. never crosses the ABI: each entry
// point picks its own, so a caller cannot ask for a mixture of the two
enum abMode { AB_EVENTS, AB_GRID };

// what ends an escape run that no basin ever absorbs: a time, or a number of
// events. unlike the two tables above these are the same table -- one row per
// particle, and the label saying where it ended -- so the choice is a flag here
// rather than a second recorder
enum abLimit { AB_UNTIL_TIME, AB_UNTIL_EVENTS };

extern "C" int abObjectCount() { return nObj; }

// did a target region actually land on the device? one empty region, asked once
// when the library is loaded.
//
// a GPU build that quietly ran on the host is the single failure this backend
// cannot be allowed to report as success: the numbers come back correct, the
// wall time is merely wrong, and nothing anywhere says which processor did the
// work. the host builds answer 0 honestly -- neither of them has a device --
// which is why only the GPU backend asks
extern "C" int abOnDevice() {
#ifdef _OPENMP
    int onHost = 1;
#pragma omp target map(from : onHost)
    { onHost = omp_is_initial_device(); }
    return !onHost;
#else
    return 0;
#endif
}

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
// this writes; n < maxRows is belt and braces against that arithmetic drifting.
//
// a basin ends the run wherever it falls, so its row is kept whatever the
// stride: it is the one row that is not a sample of a continuing trajectory but
// the end of one, and a record that stopped without saying where would be a
// record of nothing in particular
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

        if (kind == EV_ABSORB) { // the particle left: s is on the basin
            if (n < maxRows) {
                abWriteRow(o, i, n, t, s, kind);
                ++n;
            }
            break;
        }
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
        if (next == EV_ABSORB) break;        // already coasted out the rest.
        t = tEnd;                            // a basin ends the run, so the
        kind = next;                         // grid stops there: counts says
    }                                        // how far it got
    return n;
}

// one particle, run until a basin absorbs it or until the run's limit is spent,
// whichever comes first. only the ending is kept: no row is written per event,
// whichever limit is counting them, so this costs the same memory for a run of
// one collision and a run of a million.
//
// escaping is the same either way, and the difference is the whole point of the
// function: hitting a basin reports the state AT the surface, velocity still
// pointing into it, because that is the escape being measured -- responding to
// the surface would describe a bounce that never happened.
//
// not escaping is what the limit decides. AB_UNTIL_TIME coasts the last flight
// to exactly tf and reports basin 0 there; AB_UNTIL_EVENTS has no tf to coast
// to, so it reports basin 0 at the last event it took, and maxEvents is the
// count asked for rather than a guard against one particle running away
static void abEscapeOne(state s, double tf, int maxEvents, int limit, int i,
                        const abEscape *o) {
    const int cols = o->columns;
    double t = 0.0;
    int kind = EV_INITIAL, basin = 0, absorbed = 0;

    // every way out has to leave t and basin agreeing: absorbed (basin > 0,
    // t = when), or basin 0 at whatever the limit means -- t = tf exactly for a
    // timed run, the last event's time for an event-limited one. A timed run
    // spending maxEvents instead is the Zeno guard firing, and reports as the
    // latter: the one case where basin 0 does not mean t == tf
    for (int ev = 0; ev < maxEvents; ++ev) {
        state s0 = s;
        double tc = stepScene(&s, objs, nObj, &kind, &basin);

        // nothing ahead, or nothing ahead before tf: coast the parabola from
        // where the flight began and stop. s may have been advanced past tf by
        // the step just taken, which is why s0 is the one that is used
        if (limit == AB_UNTIL_TIME && (tc == MAGIC_NO_COLLISION || t + tc > tf)) {
            s = trajectory(s0, tf - t);
            t = tf;
            break;
        }

        // the same emptiness with no tf to fill: counting events, a particle
        // with nothing left to hit will never reach another one, so the run is
        // over at the last event it did reach. stepScene leaves s where it was
        if (tc == MAGIC_NO_COLLISION) break;

        t += tc;
        if (kind == EV_ABSORB) { // s is already snapped onto the basin
            absorbed = 1;
            break;
        }
    }
    if (!absorbed) basin = 0; // stepScene only writes it when one is hit

    if (cols & AB_T)      o->t[i]      = t;
    if (cols & AB_X)      o->x[i]      = s.x;
    if (cols & AB_Y)      o->y[i]      = s.y;
    if (cols & AB_VX)     o->vx[i]     = s.vx;
    if (cols & AB_VY)     o->vy[i]     = s.vy;
    if (cols & AB_ENERGY) o->energy[i] = 0.5 * (s.vx * s.vx + s.vy * s.vy) + g * s.y;
    if (cols & AB_BASIN)  o->basin[i]  = basin;
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

// the escape run keeps one row per particle instead of a table, so its map
// clauses are all [0:nPart] and it does not share abRun's block above
static int abRunEscape(const double *x0, const double *y0,
                       const double *vx0, const double *vy0,
                       int nPart, double tf, int maxEvents, int limit,
                       abEscape *o) {
    const int cols = o->columns;
    double *ot = o->t, *ox = o->x, *oy = o->y;
    double *ovx = o->vx, *ovy = o->vy, *oe = o->energy;
    int *ob = o->basin;

    // sized to 1 for any column the caller did not ask for: OpenMP cannot map
    // a null pointer, so the binding passes a throwaway and leaves the bit clear
    const int lt  = (cols & AB_T)      ? nPart : 1;
    const int lx  = (cols & AB_X)      ? nPart : 1;
    const int ly  = (cols & AB_Y)      ? nPart : 1;
    const int lvx = (cols & AB_VX)     ? nPart : 1;
    const int lvy = (cols & AB_VY)     ? nPart : 1;
    const int le  = (cols & AB_ENERGY) ? nPart : 1;
    const int lb  = (cols & AB_BASIN)  ? nPart : 1;

#ifdef _OPENMP
#pragma omp target teams distribute parallel for                            \
    map(to : x0[0 : nPart], y0[0 : nPart], vx0[0 : nPart], vy0[0 : nPart])  \
    map(from : ot[0 : lt], ox[0 : lx], oy[0 : ly], ovx[0 : lvx],            \
               ovy[0 : lvy], oe[0 : le], ob[0 : lb])
#endif
    for (int i = 0; i < nPart; ++i) {
        abEscape dev = {ot, ox, oy, ovx, ovy, oe, ob, cols};
        abEscapeOne({x0[i], y0[i], vx0[i], vy0[i]}, tf, maxEvents, limit, i, &dev);
    }
    return 0;
}

// one entry point per limit, as the recorders do: an escape run gives up either
// at a time or after a number of collisions, and no single argument list could
// say which of the two a caller meant. Both fill the same one row per particle.
//
// maxEvents is the Zeno guard here and nothing else: a scene that traps a
// particle in unboundedly many collisions before tf would otherwise never
// finish, and the guard makes it report basin 0 early instead
extern "C" int abRunBasinsTime(const double *x0, const double *y0,
                               const double *vx0, const double *vy0,
                               int nPart, double tf, int maxEvents,
                               abEscape *o) {
    if (nPart < 1 || maxEvents < 1) return 1;
    if (!(tf >= 0.0)) return 1; // also catches NaN
    return abRunEscape(x0, y0, vx0, vy0, nPart, tf, maxEvents, AB_UNTIL_TIME, o);
}

// here the count IS the limit, so there is no tf to pass and none to reach: a
// particle that survives `iterations` events reports basin 0 at the last of
// them. 0 is allowed and means exactly what it says -- the initial state, which
// no basin has had a chance to absorb yet
extern "C" int abRunBasinsIterations(const double *x0, const double *y0,
                                     const double *vx0, const double *vy0,
                                     int nPart, int iterations, abEscape *o) {
    if (nPart < 1 || iterations < 0) return 1;
    return abRunEscape(x0, y0, vx0, vy0, nPart, 0.0, iterations,
                       AB_UNTIL_EVENTS, o);
}
