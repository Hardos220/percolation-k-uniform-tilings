/* Newman-Ziff site/bond percolation on an Lx x Ly torus of a periodic cell graph.
 *
 * For every sample the sites (or bonds) are occupied in random order and the
 * occupation numbers n_h, n_v at which a cluster first wraps along T1 ("h")
 * and along T2 ("v") are recorded.  A cluster wraps when two of its sites are
 * joined by a path whose unwrapped displacement (in cell units) is a nonzero
 * multiple of (Lx, Ly); this is detected with union-find that stores, for every
 * site, the displacement to its parent (Newman & Ziff, PRE 64, 016706, 2001).
 * Every root also keeps the first winding vector found in its cluster; a second,
 * non-parallel winding means a cross cluster (homology Z x Z), occupation n_x.
 * A sample stops at n_x (then n_h and n_v are known too).
 *
 * Output (text): sparse histograms over n of n_h, n_v, min(n_h,n_v), max(n_h,n_v), n_x.
 *
 * usage: nz CELLFILE site|bond Lx Ly SAMPLES SEED OUTFILE [MAX_CPU_SECONDS]
 * CELLFILE: "nv ne" then ne lines "u w da db" (edge from orbit u of cell (0,0)
 *           to orbit w of cell (da,db)).
 * With MAX_CPU_SECONDS the run stops early; the header gives the samples done.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double cpu_seconds(void) {
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    ULARGE_INTEGER x;
    x.LowPart = u.dwLowDateTime; x.HighPart = u.dwHighDateTime;
    return (double)x.QuadPart * 1e-7;
}
#else
#include <time.h>
static double cpu_seconds(void) { return (double)clock() / CLOCKS_PER_SEC; }
#endif

static uint64_t rs[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t next64(void) { /* xoshiro256** */
    uint64_t r = rotl(rs[1] * 5, 7) * 9, t = rs[1] << 17;
    rs[2] ^= rs[0]; rs[3] ^= rs[1]; rs[1] ^= rs[2]; rs[0] ^= rs[3];
    rs[2] ^= t; rs[3] = rotl(rs[3], 45);
    return r;
}
static void seed_rng(uint64_t s) {
    for (int i = 0; i < 4; i++) { /* splitmix64 */
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rs[i] = z ^ (z >> 31);
    }
}
static inline uint32_t bounded(uint32_t n) { /* Lemire, unbiased */
    uint64_t m = (uint64_t)(uint32_t)(next64() >> 32) * n;
    uint32_t l = (uint32_t)m;
    if (l < n) {
        uint32_t t = (uint32_t)(-n) % n;
        while (l < t) { m = (uint64_t)(uint32_t)(next64() >> 32) * n; l = (uint32_t)m; }
    }
    return (uint32_t)(m >> 32);
}

static int32_t *par, *ddx, *ddy; /* parent (<0: root, -size), displacement to parent */
static int32_t *wnx, *wny;        /* first winding vector of the cluster (at its root) */

/* record winding (wx, wy) in cluster with root r; returns 4 if it makes a cross */
static inline int add_winding(int32_t r, int32_t wx, int32_t wy) {
    if (wnx[r] == 0 && wny[r] == 0) { wnx[r] = wx; wny[r] = wy; return 0; }
    return ((int64_t)wnx[r] * wy - (int64_t)wny[r] * wx) != 0 ? 4 : 0;
}

static inline int32_t find(int32_t i, int32_t *tx, int32_t *ty) {
    int32_t sx = 0, sy = 0;
    while (par[i] >= 0) { /* path halving with displacement bookkeeping */
        int32_t p = par[i];
        if (par[p] >= 0) { ddx[i] += ddx[p]; ddy[i] += ddy[p]; par[i] = par[p]; }
        sx += ddx[i]; sy += ddy[i];
        i = par[i];
    }
    *tx = sx; *ty = sy;
    return i;
}

/* join s and t, where t = s + (ex, ey) in unwrapped cell coordinates.
 * returns flags: bit0 = winding along T1, bit1 = along T2, bit2 = cross */
static inline int join(int32_t s, int32_t t, int32_t ex, int32_t ey) {
    int32_t ax, ay, bx, by;
    int32_t r1 = find(s, &ax, &ay), r2 = find(t, &bx, &by);
    int32_t wx = ax - ex - bx, wy = ay - ey - by; /* pos(r1) - pos(r2) */
    if (r1 == r2) {
        if (wx == 0 && wy == 0) return 0;
        return (wx != 0) | ((wy != 0) << 1) | add_winding(r1, wx, wy);
    }
    int32_t big = r1, small = r2;
    if (par[r1] <= par[r2]) { /* r1 larger: attach r2 under r1 */
        par[r1] += par[r2]; par[r2] = r1; ddx[r2] = wx; ddy[r2] = wy;
    } else {
        big = r2; small = r1;
        par[r2] += par[r1]; par[r1] = r2; ddx[r1] = -wx; ddy[r1] = -wy;
    }
    if (wnx[small] != 0 || wny[small] != 0) return add_winding(big, wnx[small], wny[small]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 8 && argc != 9) {
        fprintf(stderr, "usage: nz CELLFILE site|bond Lx Ly SAMPLES SEED OUTFILE [MAX_CPU_SECONDS]\n");
        return 1;
    }
    double tmax = argc == 9 ? atof(argv[8]) : 0;
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    int nv, ne;
    if (fscanf(f, "%d %d", &nv, &ne) != 2) return 1;
    int *eu = malloc(ne * sizeof(int)), *ew = malloc(ne * sizeof(int));
    int *ea = malloc(ne * sizeof(int)), *eb = malloc(ne * sizeof(int));
    for (int k = 0; k < ne; k++)
        if (fscanf(f, "%d %d %d %d", &eu[k], &ew[k], &ea[k], &eb[k]) != 4) return 1;
    fclose(f);
    int bond = strcmp(argv[2], "bond") == 0;
    int Lx = atoi(argv[3]), Ly = atoi(argv[4]);
    long long M = atoll(argv[5]);
    seed_rng(strtoull(argv[6], NULL, 10));
    int32_t N = (int32_t)Lx * Ly * nv;
    int64_t NE = (int64_t)Lx * Ly * ne;

    /* edge list of the torus */
    int32_t *Es = malloc(NE * sizeof(int32_t)), *Et = malloc(NE * sizeof(int32_t));
    int8_t *Ex = malloc(NE), *Ey = malloc(NE);
    int64_t m = 0;
    for (int b = 0; b < Ly; b++)
        for (int a = 0; a < Lx; a++)
            for (int k = 0; k < ne; k++) {
                int a2 = ((a + ea[k]) % Lx + Lx) % Lx, b2 = ((b + eb[k]) % Ly + Ly) % Ly;
                Es[m] = ((int32_t)b * Lx + a) * nv + eu[k];
                Et[m] = ((int32_t)b2 * Lx + a2) * nv + ew[k];
                Ex[m] = (int8_t)ea[k]; Ey[m] = (int8_t)eb[k];
                m++;
            }
    /* CSR adjacency for site percolation */
    int32_t *deg = calloc(N + 1, sizeof(int32_t)), *nb = NULL;
    int8_t *nbx = NULL, *nby = NULL;
    if (!bond) {
        for (int64_t k = 0; k < NE; k++) { deg[Es[k] + 1]++; deg[Et[k] + 1]++; }
        for (int32_t i = 0; i < N; i++) deg[i + 1] += deg[i];
        nb = malloc(2 * NE * sizeof(int32_t)); nbx = malloc(2 * NE); nby = malloc(2 * NE);
        int32_t *fill = malloc(N * sizeof(int32_t));
        memcpy(fill, deg, N * sizeof(int32_t));
        for (int64_t k = 0; k < NE; k++) {
            int32_t s = Es[k], t = Et[k];
            nb[fill[s]] = t; nbx[fill[s]] = Ex[k]; nby[fill[s]] = Ey[k]; fill[s]++;
            nb[fill[t]] = s; nbx[fill[t]] = -Ex[k]; nby[fill[t]] = -Ey[k]; fill[t]++;
        }
        free(fill);
    }
    int64_t Nobj = bond ? NE : N;
    int32_t *order = malloc(Nobj * sizeof(int32_t));
    for (int64_t i = 0; i < Nobj; i++) order[i] = (int32_t)i;
    par = malloc(N * sizeof(int32_t)); ddx = calloc(N, sizeof(int32_t)); ddy = calloc(N, sizeof(int32_t));
    wnx = calloc(N, sizeof(int32_t)); wny = calloc(N, sizeof(int32_t));
    uint8_t *occ = calloc(N, 1);
    const int32_t EMPTY = INT32_MIN;
    for (int32_t i = 0; i < N; i++) par[i] = bond ? -1 : EMPTY;

    int64_t *hh = calloc(Nobj + 1, sizeof(int64_t)), *hv = calloc(Nobj + 1, sizeof(int64_t));
    int64_t *hmin = calloc(Nobj + 1, sizeof(int64_t)), *hmax = calloc(Nobj + 1, sizeof(int64_t));
    int64_t *hx = calloc(Nobj + 1, sizeof(int64_t));
    const char *pf = getenv("RMQ_PROGRESS_FILE");
    long long report = M / 200 > 0 ? M / 200 : 1;

    long long smp;
    for (smp = 0; smp < M; smp++) {
        if (tmax > 0 && cpu_seconds() > tmax) break;
        int64_t nh = -1, nvv = -1, nx = -1, n;
        for (n = 0; n < Nobj; n++) {
            uint32_t j = (uint32_t)n + bounded((uint32_t)(Nobj - n)); /* incremental Fisher-Yates */
            int32_t x = order[j]; order[j] = order[n]; order[n] = x;
            int w = 0;
            if (bond) {
                w = join(Es[x], Et[x], Ex[x], Ey[x]);
            } else {
                par[x] = -1; ddx[x] = 0; ddy[x] = 0; wnx[x] = 0; wny[x] = 0; occ[x] = 1;
                for (int32_t q = deg[x]; q < deg[x + 1]; q++)
                    if (occ[nb[q]]) w |= join(x, nb[q], nbx[q], nby[q]);
            }
            if (w) {
                if ((w & 1) && nh < 0) nh = n + 1;
                if ((w & 2) && nvv < 0) nvv = n + 1;
                if (w & 4) { nx = n + 1; n++; break; }
            }
        }
        /* reset state touched by this sample */
        if (bond) {
            for (int64_t k = 0; k < n; k++) {
                int32_t x = order[k];
                par[Es[x]] = -1; ddx[Es[x]] = 0; ddy[Es[x]] = 0; wnx[Es[x]] = 0; wny[Es[x]] = 0;
                par[Et[x]] = -1; ddx[Et[x]] = 0; ddy[Et[x]] = 0; wnx[Et[x]] = 0; wny[Et[x]] = 0;
            }
        } else {
            for (int64_t k = 0; k < n; k++) { int32_t x = order[k]; occ[x] = 0; par[x] = EMPTY; }
        }
        hh[nh]++; hv[nvv]++;
        hmin[nh < nvv ? nh : nvv]++; hmax[nh > nvv ? nh : nvv]++;
        if (nx < 0) { fprintf(stderr, "sample %lld: no cross at full occupation\n", smp); return 2; }
        hx[nx]++;
        if (pf && (smp + 1) % report == 0) {
            FILE *g = fopen(pf, "w");
            if (g) { fprintf(g, "%lld/%lld samples\n", smp + 1, M); fclose(g); }
        }
    }
    FILE *o = fopen(argv[7], "w");
    if (!o) { perror(argv[7]); return 1; }
    fprintf(o, "# cell %s mode %s Lx %d Ly %d nv %d ne %d N %lld samples %lld seed %s\n",
            argv[1], bond ? "bond" : "site", Lx, Ly, nv, ne, (long long)Nobj, smp, argv[6]);
    fprintf(o, "# n count_h count_v count_min count_max count_cross\n");
    for (int64_t k = 0; k <= Nobj; k++)
        if (hh[k] | hv[k] | hmin[k] | hmax[k] | hx[k])
            fprintf(o, "%lld %lld %lld %lld %lld %lld\n", (long long)k, (long long)hh[k], (long long)hv[k],
                    (long long)hmin[k], (long long)hmax[k], (long long)hx[k]);
    fclose(o);
    return 0;
}
