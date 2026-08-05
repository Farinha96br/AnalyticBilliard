// long-run stability: does the billiard still behave after 1e8 bounces?
//
//   ./runStability.sh          (or: nvc++ -O2 -Mnofma -mp=multicore testStability.cpp)
//
// 1e8 bounces per particle is 4 GB of Row each, so nothing is kept. Rather than
// write a second copy of the collision loop to avoid that, this drives the real
// getCollisionStates in chunks and reduces each buffer before dropping it. The
// loop carries no state but the particle's own, so restarting a chunk from the
// previous chunk's last state continues it exactly -- the only thing the driver
// has to carry across is the elapsed time.
#include "physics.h"
#include <cstdio>
#include <cstdlib>

static const long long TOTAL = 100000000; // bounces per particle
static const int CHUNK = 1000000;         // per getCollisionStates call
static const int NPART = 5;

// where the run is inspected. the point is the growth rate, not any one number:
// round-off alone random walks like sqrt(N), while a real bias in the bounce
// grows like N
static const long long CHECK[] = {10000, 100000, 1000000, 10000000, 100000000};
static const int NCHECK = 5;

struct Stats {
    double maxdE[NCHECK];   // |E - E0| at each checkpoint
    double timeErr[NCHECK]; // naive elapsed time vs a Kahan-compensated sum
    double minDt;           // smallest gap between collisions
    long long tinyDt;       // how many were within 10*deadTime of zero
    long long escapes;      // left the box
    long long inElipse;     // got inside the solid obstacle
    long long offArc;       // touched the arc's ellipse outside the arc itself
    long long backwards;    // time failed to increase
    long long bounces;      // how many actually happened, vs TOTAL
};

int main() {
    const int nObj = 7;
    Object objs[nObj] = {
        makeLine(0.0, 1.0, 0.0),
        makeLine(0.0, 1.0, -10.0),
        makeLine(1.0, 0.0, 20.0),
        makeLine(1.0, 0.0, -20.0),
        makeLineSegment(0.0, 0.0, 10.0, 5.0),
        makeElipse(-11.0, 5.0, 5.0, 2.0, 0.5),
        makeElipseArc(15.0, 5.0, 4.0, 2.5, -0.3, 3.4, 6.0),
    };
    Object ell = objs[5], arc = objs[6];

    // exactly Box.cpp's initial conditions, srand left uncalled so the default
    // seed matches. it is worth being deliberate about this: particle 0 starts at
    // y = 0, sitting on the floor, and only stays in the box because its velocity
    // happens to point up. seed it differently and it starts on a surface heading
    // out through it, where the deadTime filter discards the t = 0 root and there
    // is no collision left to find. a starting point on a boundary is degenerate;
    // the geometry cannot rescue it
    state init[NPART];
    for (int i = 0; i < NPART; ++i) {
        init[i] = {0.0, double(i) * 1.0,
                   (rand() % 100) / 10.0 - 5.0, (rand() % 100) / 10.0 - 5.0};
    }

    Stats st[NPART];
    long long nChunks = TOTAL / CHUNK;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < NPART; ++i) {
        Row *rows = new Row[CHUNK + 1];
        Stats &S = st[i];
        for (int c = 0; c < NCHECK; ++c) { S.maxdE[c] = 0.0; S.timeErr[c] = 0.0; }
        S.minDt = 1e300;
        S.tinyDt = S.escapes = S.inElipse = S.offArc = S.backwards = S.bounces = 0;

        state s = init[i];
        double E0 = 0.5 * (s.vx * s.vx + s.vy * s.vy) + g * s.y;
        double running = 0.0;   // as the program itself accumulates it
        double kahan = 0.0, comp = 0.0; // the same sum, error compensated
        double worst = 0.0;
        int check = 0;

        for (long long c = 0; c < nChunks; ++c) {
            int n = getCollisionStates(s, objs, nObj, CHUNK, rows);
            if (n < CHUNK + 1) break; // hit MAGIC_NO_COLLISION and gave up early

            for (int k = 1; k < n; ++k) {
                Row w = rows[k];
                double dt = w.t - rows[k - 1].t;
                if (dt < S.minDt) S.minDt = dt;
                if (dt <= 10.0 * deadTime) ++S.tinyDt;
                if (dt <= 0.0) ++S.backwards;

                running += dt;
                double y = dt - comp; // Kahan: keep the bits the sum drops
                double tt = kahan + y;
                comp = (tt - kahan) - y;
                kahan = tt;

                double E = 0.5 * (w.vx * w.vx + w.vy * w.vy) + g * w.y;
                if (fabs(E - E0) > worst) worst = fabs(E - E0);

                if (w.y < -1e-9 || w.y > 10.0 + 1e-9 ||
                    w.x < -20.0 - 1e-9 || w.x > 20.0 + 1e-9) ++S.escapes;

                // inside the solid ellipse is a place no particle may reach
                vector2D le = elipseLocal(ell, w.x, w.y);
                double G = le.x * le.x / 25.0 + le.y * le.y / 4.0;
                if (G < 1.0 - 1e-9) ++S.inElipse;

                // and anything sitting on the arc's ellipse must be on the arc
                vector2D la = elipseLocal(arc, w.x, w.y);
                double Ga = la.x * la.x / 16.0 + la.y * la.y / 6.25;
                if (fabs(Ga - 1.0) < 1e-9) {
                    state at = {w.x, w.y, w.vx, w.vy};
                    if (!onArc(arc, 0.0, at)) ++S.offArc;
                }

                ++S.bounces;
                while (check < NCHECK && S.bounces == CHECK[check]) {
                    S.maxdE[check] = worst;
                    S.timeErr[check] = fabs(running - kahan);
                    ++check;
                }
            }
            s = {rows[n - 1].x, rows[n - 1].y, rows[n - 1].vx, rows[n - 1].vy};
        }
        delete[] rows;
    }

    printf("%lld bounces x %d particles, scene of %d objects\n\n", TOTAL, NPART, nObj);
    printf("energy drift, max|E-E0| after N bounces\n");
    printf("%9s", "particle");
    for (int c = 0; c < NCHECK; ++c) printf("%13.0e", (double)CHECK[c]);
    printf("\n");
    for (int i = 0; i < NPART; ++i) {
        printf("%9d", i);
        for (int c = 0; c < NCHECK; ++c) printf("%13.3e", st[i].maxdE[c]);
        printf("\n");
    }

    printf("\nrecorded-time error, |naive sum - Kahan sum|\n");
    printf("%9s", "particle");
    for (int c = 0; c < NCHECK; ++c) printf("%13.0e", (double)CHECK[c]);
    printf("\n");
    for (int i = 0; i < NPART; ++i) {
        printf("%9d", i);
        for (int c = 0; c < NCHECK; ++c) printf("%13.3e", st[i].timeErr[c]);
        printf("\n");
    }

    printf("\n%9s %12s %12s %10s %10s %10s %10s %10s\n", "particle", "bounces",
           "min dt", "tiny dt", "escapes", "in elipse", "off arc", "dt<=0");
    int bad = 0;
    for (int i = 0; i < NPART; ++i) {
        Stats &S = st[i];
        printf("%9d %12lld %12.3e %10lld %10lld %10lld %10lld %10lld\n", i, S.bounces,
               S.minDt, S.tinyDt, S.escapes, S.inElipse, S.offArc, S.backwards);
        if (S.bounces != TOTAL || S.escapes || S.inElipse || S.offArc || S.backwards) ++bad;
    }
    printf("\n%s\n", bad ? "FAILED: see the non-zero columns above"
                         : "all particles completed, nothing escaped, no stalls");
    return bad ? 1 : 0;
}
