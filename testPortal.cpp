// portal tests. build twice, the torus section needs gravity switched off:
//   g++ -O2 -ffp-contract=off -o tp.out  testPortal.cpp
//   g++ -O2 -ffp-contract=off -DGRAVITY=0.0 -o tp0.out testPortal.cpp
#include "physics.h"
#include <cstdio>

static int fails = 0;
static void ok(const char *what, bool good, double detail = 0.0) {
    printf("  %-52s %s", what, good ? "ok" : "FAIL");
    if (!good) { printf("   (%.17g)", detail); ++fails; }
    printf("\n");
}

int main() {
    // ---- 1. the pair is forced to a matching size, and maps distance ----
    printf("=== equal size enforced, 4-long entry ===\n");
    {
        // the exit is asked for at length 12; it must come back rebuilt at 4,
        // keeping only the direction it was pointed in
        Object a = asPortal(makeLineSegment(0.0, 0.0, 0.0, 4.0),
                            10.0, 0.0, 10.0, 12.0, 1.0);
        double ex = a.q[6] - a.q[4], ey = a.q[7] - a.q[5];
        ok("an over-long exit is rebuilt at the entry's length",
           fabs(sqrt(ex * ex + ey * ey) - 4.0) < 1e-12, sqrt(ex * ex + ey * ey));
        ok("and keeps the direction it was given", fabs(ex) < 1e-12 && ey > 0.0, ex);

        // with matching sizes, distance in is distance out and ends stay ends
        double at[3] = {0.0, 2.0, 4.0};
        for (int i = 0; i < 3; ++i) {
            state s = {0.0, at[i], 1.0, 0.0};
            state r = teleport(a, s);
            char msg[80];
            snprintf(msg, sizeof msg, "enter %.1f along -> exit %.1f along", at[i], at[i]);
            ok(msg, fabs(r.y - at[i]) < 1e-12 && fabs(r.x - 10.0) < 1e-12, r.y);
        }

        Object b = asPortal(makeLineSegment(0.0, 0.0, 0.0, 4.0),
                            10.0, 1.0, 10.0, 5.0, 1.0);
        ok("an exit already the right size is untouched",
           b.q[6] == 10.0 && b.q[7] == 5.0, b.q[7]);
    }

    // ---- 2. a quarter turn: upright entry, flat exit, as Portals.cpp uses ----
    // the answer is exact here, which is the point of picking 90 degrees. with
    // tA = (0,1) and nA = (-1,0) against tB = (1,0) and nB = (0,1), flip = +1
    // sends (vx, vy) to (vy, -vx): a particle arriving level leaves falling
    printf("\n=== quarter turn, upright -> flat ===\n");
    {
        Object a = asPortal(makeLineSegment(-8.0, 2.0, -8.0, 8.0),
                            2.0, 5.0, 8.0, 5.0, 1.0);
        double ax = a.p[2] - a.p[0], ay = a.p[3] - a.p[1];
        double bx = a.q[6] - a.q[4], by = a.q[7] - a.q[5];
        ok("both ends the same length (6)",
           fabs(sqrt(ax * ax + ay * ay) - 6.0) < 1e-12 &&
           fabs(sqrt(bx * bx + by * by) - 6.0) < 1e-12, sqrt(bx * bx + by * by));
        ok("the exit endpoint lands exactly where asked",
           a.q[6] == 8.0 && a.q[7] == 5.0, a.q[6]);

        state atStart = {-8.0, 2.0, 1.0, 0.0};
        state atEnd = {-8.0, 8.0, 1.0, 0.0};
        state rs = teleport(a, atStart), re = teleport(a, atEnd);
        ok("the start maps to the exit's start (2,5)",
           fabs(rs.x - 2.0) < 1e-12 && fabs(rs.y - 5.0) < 1e-12, rs.x);
        ok("the end maps to the exit's end (8,5)",
           fabs(re.x - 8.0) < 1e-12 && fabs(re.y - 5.0) < 1e-12, re.x);

        // the middle of one lands on the middle of the other
        state mid = {-8.0, 5.0, 1.0, 0.0};
        state rm = teleport(a, mid);
        ok("the middle maps to the middle (5,5)",
           fabs(rm.x - 5.0) < 1e-12 && fabs(rm.y - 5.0) < 1e-12, rm.x);

        state s = {-8.0, 4.5, 2.0, 0.5};
        state r = teleport(a, s);
        ok("(vx, vy) comes out as (vy, -vx)",
           fabs(r.vx - s.vy) < 1e-14 && fabs(r.vy + s.vx) < 1e-14, r.vx);
        ok("arriving level, leaving straight down",
           fabs(teleport(a, (state){-8.0, 5.0, 3.0, 0.0}).vy + 3.0) < 1e-14);
        double v0 = sqrt(s.vx * s.vx + s.vy * s.vy);
        double v1 = sqrt(r.vx * r.vx + r.vy * r.vy);
        ok("speed survives the turn", fabs(v0 - v1) < 1e-14, v1 - v0);
    }

    // ---- 3. speed is untouched, both flips ----
    printf("\n=== speed preserved ===\n");
    {
        Object p = asPortal(makeLineSegment(0.0, 0.0, 0.0, 4.0), 3.0, 7.0, 9.0, 1.0, 1.0);
        Object m = asPortal(makeLineSegment(0.0, 0.0, 0.0, 4.0), 3.0, 7.0, 9.0, 1.0, -1.0);
        state s = {0.0, 1.7, 0.83, -2.4};
        double v0 = sqrt(s.vx * s.vx + s.vy * s.vy);
        state rp = teleport(p, s), rm = teleport(m, s);
        ok("flip = +1 leaves |v| unchanged",
           fabs(sqrt(rp.vx * rp.vx + rp.vy * rp.vy) - v0) < 1e-14);
        ok("flip = -1 leaves |v| unchanged",
           fabs(sqrt(rm.vx * rm.vx + rm.vy * rm.vy) - v0) < 1e-14);
    }

    // ---- 4. end to end: a pair facing each other and nothing else ----
    // the whole scene is two portals, so the particle can only ever cross between
    // them. without gravity it should keep going forever without its velocity or
    // its height ever changing -- any drift is the transform failing to be a
    // rigid motion, and would show up nowhere else
    printf("\n=== round trip through a facing pair (flip = +1), g = %.1f ===\n", g);
    if (g != 0.0) {
        printf("  skipped: rebuild with -DGRAVITY=0.0 for this one\n");
    } else {
        const int nObj = 2;
        Object objs[nObj] = {
            asPortal(makeLineSegment(-10.0, 0.0, -10.0, 10.0), 10.0, 0.0, 10.0, 10.0, 1.0),
            asPortal(makeLineSegment(10.0, 0.0, 10.0, 10.0), -10.0, 0.0, -10.0, 10.0, 1.0),
        };
        Row *rows = new Row[401];
        state s = {0.0, 3.0, -1.0, 0.0};
        int n = getCollisionStates(s, objs, nObj, 100, rows, 401);

        bool vsame = true, ysame = true, paired = true;
        for (int k = 0; k < n; ++k) {
            if (rows[k].vx != -1.0 || rows[k].vy != 0.0) vsame = false;
            if (rows[k].y != 3.0) ysame = false;
        }
        for (int k = 1; k + 1 < n; k += 2) {
            if (rows[k].t != rows[k + 1].t) paired = false;
            if (fabs(fabs(rows[k].x - rows[k + 1].x) - 20.0) > 1e-12) paired = false;
        }
        printf("  rows written: %d\n", n);
        ok("velocity never changes, however many crossings", vsame);
        ok("height never changes", ysame);
        ok("each teleport is an arrival and a departure at one instant", paired);
        delete[] rows;
    }

    // ---- 6. a reflecting scene must be untouched by all of this ----
    printf("\n=== the plain box still reflects exactly as before ===\n");
    {
        const int nObj = 4;
        Object objs[nObj] = {
            makeLine(0.0, 1.0, 0.0), makeLine(0.0, 1.0, -10.0),
            makeLine(1.0, 0.0, 20.0), makeLine(1.0, 0.0, -20.0),
        };
        Row *rows = new Row[201];
        state s = {0.0, 5.0, 3.3, 3.6};
        int n = getCollisionStates(s, objs, nObj, 200, rows, 201);
        double E0 = 0.5 * (s.vx * s.vx + s.vy * s.vy) + g * s.y, worst = 0.0;
        for (int k = 0; k < n; ++k) {
            double E = 0.5 * (rows[k].vx * rows[k].vx + rows[k].vy * rows[k].vy)
                       + g * rows[k].y;
            if (fabs(E - E0) > worst) worst = fabs(E - E0);
        }
        printf("  rows %d, max|E-E0| = %.3e\n", n, worst);
        ok("no portals means exactly one row per bounce", n == 201);
        ok("energy still conserved", worst < 1e-12, worst);
        delete[] rows;
    }

    printf("\n%s\n", fails ? "FAILURES ABOVE" : "all portal tests passed");
    return fails != 0;
}
