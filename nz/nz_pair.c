/* Coupled Newman-Ziff runs on the two members of a slip pair.
 *
 * The two cell files have the same number of vertices and edges, and vertex v / edge k of one correspond to
 * vertex v / edge k of the other (analysis/slip_coupled_cells.py).  On Lx x Ly tori of both cells the elements
 * (sites or bonds) are numbered identically, and every sample occupies them in one common random order on both
 * tori.  For each torus the occupation numbers at the first wrapping in any direction and at the first cross
 * configuration are recorded, as in nz.c; the output histograms have the same format as those of nz.c, so the two
 * thresholds are estimated as before, and the variance of their difference is estimated from independent runs.
 *
 * TWIST: the staggered cell has the basis (T1, T2 + s T1/2); its lattice contains Ly T2 for even Ly, so with
 * TWIST = s Ly/2 both members live on the same torus, the plane modulo <Lx T1, Ly T2>: for the staggered member a
 * cell (a, b + Ly) is identified with (a + TWIST, b).  TWIST = 0: ordinary Lx x Ly tori of both cells.
 *
 * usage: nz_pair CELL_E CELL_S site|bond Lx Ly SAMPLES SEED OUT_E OUT_S TWIST [MAX_CPU_SECONDS]
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

typedef struct {
    int nv, ne;
    int32_t N;
    int64_t NE;
    int32_t *Es, *Et;
    int8_t *Ex, *Ey;
    int32_t *deg, *nb;
    int8_t *nbx, *nby;
    int32_t *par, *ddx, *ddy, *wnx, *wny;
    uint8_t *occ;
    int64_t *hh, *hv, *hmin, *hmax, *hx;
} Torus;

static const int32_t EMPTY = INT32_MIN;

static inline int add_winding(Torus *T, int32_t r, int32_t wx, int32_t wy) {
    if (T->wnx[r] == 0 && T->wny[r] == 0) { T->wnx[r] = wx; T->wny[r] = wy; return 0; }
    return ((int64_t)T->wnx[r] * wy - (int64_t)T->wny[r] * wx) != 0 ? 4 : 0;
}

static inline int32_t find(Torus *T, int32_t i, int32_t *tx, int32_t *ty) {
    int32_t sx = 0, sy = 0;
    int32_t *par = T->par, *ddx = T->ddx, *ddy = T->ddy;
    while (par[i] >= 0) {
        int32_t p = par[i];
        if (par[p] >= 0) { ddx[i] += ddx[p]; ddy[i] += ddy[p]; par[i] = par[p]; }
        sx += ddx[i]; sy += ddy[i];
        i = par[i];
    }
    *tx = sx; *ty = sy;
    return i;
}

static inline int join(Torus *T, int32_t s, int32_t t, int32_t ex, int32_t ey) {
    int32_t ax, ay, bx, by;
    int32_t r1 = find(T, s, &ax, &ay), r2 = find(T, t, &bx, &by);
    int32_t wx = ax - ex - bx, wy = ay - ey - by;
    if (r1 == r2) {
        if (wx == 0 && wy == 0) return 0;
        return (wx != 0) | ((wy != 0) << 1) | add_winding(T, r1, wx, wy);
    }
    int32_t big = r1, small = r2;
    if (T->par[r1] <= T->par[r2]) {
        T->par[r1] += T->par[r2]; T->par[r2] = r1; T->ddx[r2] = wx; T->ddy[r2] = wy;
    } else {
        big = r2; small = r1;
        T->par[r2] += T->par[r1]; T->par[r1] = r2; T->ddx[r1] = -wx; T->ddy[r1] = -wy;
    }
    if (T->wnx[small] != 0 || T->wny[small] != 0) return add_winding(T, big, T->wnx[small], T->wny[small]);
    return 0;
}

static int load(Torus *T, const char *fn, int bond, int Lx, int Ly, int twist) {
    FILE *f = fopen(fn, "r");
    if (!f) { perror(fn); return 1; }
    if (fscanf(f, "%d %d", &T->nv, &T->ne) != 2) return 1;
    int ne = T->ne, nv = T->nv;
    int *eu = malloc(ne * sizeof(int)), *ew = malloc(ne * sizeof(int));
    int *ea = malloc(ne * sizeof(int)), *eb = malloc(ne * sizeof(int));
    for (int k = 0; k < ne; k++)
        if (fscanf(f, "%d %d %d %d", &eu[k], &ew[k], &ea[k], &eb[k]) != 4) return 1;
    fclose(f);
    T->N = (int32_t)Lx * Ly * nv;
    T->NE = (int64_t)Lx * Ly * ne;
    T->Es = malloc(T->NE * sizeof(int32_t)); T->Et = malloc(T->NE * sizeof(int32_t));
    T->Ex = malloc(T->NE); T->Ey = malloc(T->NE);
    int64_t m = 0;
    for (int b = 0; b < Ly; b++)
        for (int a = 0; a < Lx; a++)
            for (int k = 0; k < ne; k++) {
                int braw = b + eb[k];
                int jw = braw >= Ly ? 1 : (braw < 0 ? -1 : 0); /* crossings of the Ly boundary */
                int b2 = braw - jw * Ly;
                int a2 = (((a + ea[k] + jw * twist) % Lx) + Lx) % Lx;
                T->Es[m] = ((int32_t)b * Lx + a) * nv + eu[k];
                T->Et[m] = ((int32_t)b2 * Lx + a2) * nv + ew[k];
                T->Ex[m] = (int8_t)ea[k]; T->Ey[m] = (int8_t)eb[k];
                m++;
            }
    int32_t N = T->N;
    T->deg = calloc(N + 1, sizeof(int32_t));
    if (!bond) {
        for (int64_t k = 0; k < T->NE; k++) { T->deg[T->Es[k] + 1]++; T->deg[T->Et[k] + 1]++; }
        for (int32_t i = 0; i < N; i++) T->deg[i + 1] += T->deg[i];
        T->nb = malloc(2 * T->NE * sizeof(int32_t)); T->nbx = malloc(2 * T->NE); T->nby = malloc(2 * T->NE);
        int32_t *fill = malloc(N * sizeof(int32_t));
        memcpy(fill, T->deg, N * sizeof(int32_t));
        for (int64_t k = 0; k < T->NE; k++) {
            int32_t s = T->Es[k], t = T->Et[k];
            T->nb[fill[s]] = t; T->nbx[fill[s]] = T->Ex[k]; T->nby[fill[s]] = T->Ey[k]; fill[s]++;
            T->nb[fill[t]] = s; T->nbx[fill[t]] = -T->Ex[k]; T->nby[fill[t]] = -T->Ey[k]; fill[t]++;
        }
        free(fill);
    }
    T->par = malloc(N * sizeof(int32_t)); T->ddx = calloc(N, sizeof(int32_t)); T->ddy = calloc(N, sizeof(int32_t));
    T->wnx = calloc(N, sizeof(int32_t)); T->wny = calloc(N, sizeof(int32_t));
    T->occ = calloc(N, 1);
    for (int32_t i = 0; i < N; i++) T->par[i] = bond ? -1 : EMPTY;
    int64_t Nobj = bond ? T->NE : N;
    T->hh = calloc(Nobj + 1, sizeof(int64_t)); T->hv = calloc(Nobj + 1, sizeof(int64_t));
    T->hmin = calloc(Nobj + 1, sizeof(int64_t)); T->hmax = calloc(Nobj + 1, sizeof(int64_t));
    T->hx = calloc(Nobj + 1, sizeof(int64_t));
    free(eu); free(ew); free(ea); free(eb);
    return 0;
}

/* add element x (site or bond index); returns the winding flags */
static inline int add(Torus *T, int bond, int32_t x) {
    if (bond) return join(T, T->Es[x], T->Et[x], T->Ex[x], T->Ey[x]);
    int w = 0;
    T->par[x] = -1; T->ddx[x] = 0; T->ddy[x] = 0; T->wnx[x] = 0; T->wny[x] = 0; T->occ[x] = 1;
    for (int32_t q = T->deg[x]; q < T->deg[x + 1]; q++)
        if (T->occ[T->nb[q]]) w |= join(T, x, T->nb[q], T->nbx[q], T->nby[q]);
    return w;
}

static inline void reset(Torus *T, int bond, const int32_t *order, int64_t n) {
    for (int64_t k = 0; k < n; k++) {
        int32_t x = order[k];
        if (bond) {
            int32_t s = T->Es[x], t = T->Et[x];
            T->par[s] = -1; T->ddx[s] = 0; T->ddy[s] = 0; T->wnx[s] = 0; T->wny[s] = 0;
            T->par[t] = -1; T->ddx[t] = 0; T->ddy[t] = 0; T->wnx[t] = 0; T->wny[t] = 0;
        } else {
            T->occ[x] = 0; T->par[x] = EMPTY;
        }
    }
}

static int write_hist(Torus *T, const char *fn, const char *cell, int bond, int Lx, int Ly, long long smp, const char *seed, int twist) {
    int64_t Nobj = bond ? T->NE : T->N;
    FILE *o = fopen(fn, "w");
    if (!o) { perror(fn); return 1; }
    fprintf(o, "# cell %s mode %s Lx %d Ly %d nv %d ne %d N %lld samples %lld seed %s coupled twist %d\n",
            cell, bond ? "bond" : "site", Lx, Ly, T->nv, T->ne, (long long)Nobj, smp, seed, twist);
    fprintf(o, "# n count_h count_v count_min count_max count_cross\n");
    for (int64_t k = 0; k <= Nobj; k++)
        if (T->hh[k] | T->hv[k] | T->hmin[k] | T->hmax[k] | T->hx[k])
            fprintf(o, "%lld %lld %lld %lld %lld %lld\n", (long long)k, (long long)T->hh[k], (long long)T->hv[k],
                    (long long)T->hmin[k], (long long)T->hmax[k], (long long)T->hx[k]);
    fclose(o);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 11 && argc != 12) {
        fprintf(stderr, "usage: nz_pair CELL_E CELL_S site|bond Lx Ly SAMPLES SEED OUT_E OUT_S TWIST [MAX_CPU_SECONDS]\n");
        return 1;
    }
    int bond = strcmp(argv[3], "bond") == 0;
    int Lx = atoi(argv[4]), Ly = atoi(argv[5]);
    long long M = atoll(argv[6]);
    int twist = atoi(argv[10]);
    double tmax = argc == 12 ? atof(argv[11]) : 0;
    seed_rng(strtoull(argv[7], NULL, 10));
    Torus A, B;
    if (twist && (Ly % 2)) { fprintf(stderr, "TWIST needs even Ly\n"); return 1; }
    if (load(&A, argv[1], bond, Lx, Ly, 0) || load(&B, argv[2], bond, Lx, Ly, twist)) return 1;
    if (A.nv != B.nv || A.ne != B.ne) { fprintf(stderr, "cells differ in size\n"); return 1; }
    int64_t Nobj = bond ? A.NE : A.N;
    int32_t *order = malloc(Nobj * sizeof(int32_t));
    for (int64_t i = 0; i < Nobj; i++) order[i] = (int32_t)i;
    const char *pf = getenv("RMQ_PROGRESS_FILE");
    long long report = M / 200 > 0 ? M / 200 : 1;
    long long smp;
    for (smp = 0; smp < M; smp++) {
        if (tmax > 0 && (smp & 255) == 0 && cpu_seconds() > tmax) break;
        int64_t r[2][3] = {{-1, -1, -1}, {-1, -1, -1}}; /* n_h, n_v, n_x for A and B */
        Torus *TT[2] = {&A, &B};
        int done = 0;
        int64_t n;
        for (n = 0; n < Nobj && done < 2; n++) {
            uint32_t j = (uint32_t)n + bounded((uint32_t)(Nobj - n)); /* common incremental Fisher-Yates */
            int32_t x = order[j]; order[j] = order[n]; order[n] = x;
            for (int q = 0; q < 2; q++) {
                if (r[q][2] >= 0) continue;
                int w = add(TT[q], bond, x);
                if (w) {
                    if ((w & 1) && r[q][0] < 0) r[q][0] = n + 1;
                    if ((w & 2) && r[q][1] < 0) r[q][1] = n + 1;
                    if (w & 4) { r[q][2] = n + 1; done++; }
                }
            }
        }
        for (int q = 0; q < 2; q++) {
            Torus *T = TT[q];
            reset(T, bond, order, r[q][2] >= 0 ? r[q][2] : n);
            if (r[q][2] < 0) { fprintf(stderr, "sample %lld: no cross at full occupation\n", smp); return 2; }
            int64_t nh = r[q][0], nvv = r[q][1];
            T->hh[nh]++; T->hv[nvv]++;
            T->hmin[nh < nvv ? nh : nvv]++; T->hmax[nh > nvv ? nh : nvv]++;
            T->hx[r[q][2]]++;
        }
        if (pf && (smp + 1) % report == 0) {
            FILE *g = fopen(pf, "w");
            if (g) { fprintf(g, "%lld/%lld samples\n", smp + 1, M); fclose(g); }
        }
    }
    if (write_hist(&A, argv[8], argv[1], bond, Lx, Ly, smp, argv[7], 0) ||
        write_hist(&B, argv[9], argv[2], bond, Lx, Ly, smp, argv[7], twist))
        return 1;
    return 0;
}
