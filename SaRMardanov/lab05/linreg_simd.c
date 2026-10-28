#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void (*train_fn)(const double *restrict X, const double *restrict y, double *restrict w, double *b,
                         int n, int d, int iters, double eta, double lam, double *gbuf, int ld);

static inline double soft(double z, double a)
{
    return z - fmin(fmax(z, -a), a);
}

static void train_serial(const double *restrict X, const double *restrict y, double *restrict w, double *b,
                         int n, int d, int iters, double eta, double lam, double *gbuf, int ld)
{
    (void)ld;
    double *g = gbuf;
    double bb = *b, scale = eta / n, thr = eta * lam;
    for (int k = 0; k < iters; k++) {
        double gb = 0.0;
        for (int j = 0; j < d; j++)
            g[j] = 0.0;
        for (int i = 0; i < n; i++) {
            const double *x = X + (size_t)i * d;
            double s = 0.0;
            for (int j = 0; j < d; j++)
                s += x[j] * w[j];
            double e = s + bb - y[i];
            gb += e;
            for (int j = 0; j < d; j++)
                g[j] += e * x[j];
        }
        for (int j = 0; j < d; j++)
            w[j] = soft(w[j] - scale * g[j], thr);
        bb -= scale * gb;
    }
    *b = bb;
}

static void train_simd(const double *restrict X, const double *restrict y, double *restrict w, double *b,
                       int n, int d, int iters, double eta, double lam, double *gbuf, int ld)
{
    (void)ld;
    double *g = gbuf;
    double bb = *b, scale = eta / n, thr = eta * lam;
    for (int k = 0; k < iters; k++) {
        double gb = 0.0;
#pragma omp simd
        for (int j = 0; j < d; j++)
            g[j] = 0.0;
        for (int i = 0; i < n; i++) {
            const double *x = X + (size_t)i * d;
            double s = 0.0;
#pragma omp simd reduction(+:s)
            for (int j = 0; j < d; j++)
                s += x[j] * w[j];
            double e = s + bb - y[i];
            gb += e;
#pragma omp simd
            for (int j = 0; j < d; j++)
                g[j] += e * x[j];
        }
#pragma omp simd
        for (int j = 0; j < d; j++)
            w[j] = soft(w[j] - scale * g[j], thr);
        bb -= scale * gb;
    }
    *b = bb;
}

static void train_omp(const double *restrict X, const double *restrict y, double *restrict w, double *b,
                      int n, int d, int iters, double eta, double lam, double *gbuf, int ld)
{
    double bb = *b, scale = eta / n, thr = eta * lam;
#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        double *g = gbuf + (size_t)omp_get_thread_num() * ld;
        for (int k = 0; k < iters; k++) {
            double bk = bb, gb = 0.0;
            for (int j = 0; j < d; j++)
                g[j] = 0.0;
#pragma omp for schedule(static) nowait
            for (int i = 0; i < n; i++) {
                const double *x = X + (size_t)i * d;
                double s = 0.0;
                for (int j = 0; j < d; j++)
                    s += x[j] * w[j];
                double e = s + bk - y[i];
                gb += e;
                for (int j = 0; j < d; j++)
                    g[j] += e * x[j];
            }
            g[d] = gb;
#pragma omp barrier
#pragma omp for schedule(static)
            for (int j = 0; j <= d; j++) {
                double s = 0.0;
                for (int t = 0; t < nt; t++)
                    s += gbuf[(size_t)t * ld + j];
                if (j < d)
                    w[j] = soft(w[j] - scale * s, thr);
                else
                    bb = bk - scale * s;
            }
        }
    }
    *b = bb;
}

static void train_omp_simd(const double *restrict X, const double *restrict y, double *restrict w, double *b,
                           int n, int d, int iters, double eta, double lam, double *gbuf, int ld)
{
    double bb = *b, scale = eta / n, thr = eta * lam;
#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        double *g = gbuf + (size_t)omp_get_thread_num() * ld;
        for (int k = 0; k < iters; k++) {
            double bk = bb, gb = 0.0;
#pragma omp simd
            for (int j = 0; j < d; j++)
                g[j] = 0.0;
#pragma omp for schedule(static) nowait
            for (int i = 0; i < n; i++) {
                const double *x = X + (size_t)i * d;
                double s = 0.0;
#pragma omp simd reduction(+:s)
                for (int j = 0; j < d; j++)
                    s += x[j] * w[j];
                double e = s + bk - y[i];
                gb += e;
#pragma omp simd
                for (int j = 0; j < d; j++)
                    g[j] += e * x[j];
            }
            g[d] = gb;
#pragma omp barrier
#pragma omp for schedule(static)
            for (int j = 0; j <= d; j++) {
                double s = 0.0;
                for (int t = 0; t < nt; t++)
                    s += gbuf[(size_t)t * ld + j];
                if (j < d)
                    w[j] = soft(w[j] - scale * s, thr);
                else
                    bb = bk - scale * s;
            }
        }
    }
    *b = bb;
}

static uint64_t rng_state;

static double rnd(void)
{
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) * 0x1.0p-53;
}

static void generate(double *X, double *y, int n, int d, uint64_t seed)
{
    rng_state = seed;
    double *wt = malloc((size_t)d * sizeof(double));
    for (int j = 0; j < d; j++)
        wt[j] = j % 4 == 0 ? 4.0 * rnd() - 2.0 : 0.0;
    double bt = 0.5;
    for (int i = 0; i < n; i++) {
        double *x = X + (size_t)i * d;
        double s = bt;
        for (int j = 0; j < d; j++) {
            x[j] = 2.0 * rnd() - 1.0;
            s += x[j] * wt[j];
        }
        y[i] = s + 0.1 * (2.0 * rnd() - 1.0);
    }
    free(wt);
}

static double loss(const double *X, const double *y, const double *w, double b, int n, int d, double lam)
{
    double sq = 0.0, l1 = 0.0;
    for (int i = 0; i < n; i++) {
        const double *x = X + (size_t)i * d;
        double s = 0.0;
        for (int j = 0; j < d; j++)
            s += x[j] * w[j];
        double e = s + b - y[i];
        sq += e * e;
    }
    for (int j = 0; j < d; j++)
        l1 += fabs(w[j]);
    return sq / (2.0 * n) + lam * l1;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void *alloc64(size_t bytes)
{
    return aligned_alloc(64, (bytes + 63) / 64 * 64);
}

int main(int argc, char *argv[])
{
    int n = 100000, d = 32, iters = 100, threads = 0, reps = 5;
    double lam = 1e-3, eta = 0.5, eps = 1e-6, tol = 1e-9;
    uint64_t seed = 42;
    int opt;

    while ((opt = getopt(argc, argv, "n:d:k:l:e:t:r:s:z:")) != -1) {
        switch (opt) {
        case 'n': n = atoi(optarg); break;
        case 'd': d = atoi(optarg); break;
        case 'k': iters = atoi(optarg); break;
        case 'l': lam = atof(optarg); break;
        case 'e': eta = atof(optarg); break;
        case 't': threads = atoi(optarg); break;
        case 'r': reps = atoi(optarg); break;
        case 's': seed = strtoull(optarg, NULL, 10); break;
        case 'z': eps = atof(optarg); break;
        default:
            fprintf(stderr, "Usage: %s [-n N] [-d d] [-k K] [-l lambda] [-e eta] [-t threads] [-r reps] [-s seed] [-z eps]\n",
                    argv[0]);
            return 1;
        }
    }
    if (n <= 0 || d <= 0 || iters <= 0 || reps <= 0 || lam < 0.0 || eta <= 0.0 || threads < 0) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }
    if (threads > 0)
        omp_set_num_threads(threads);
    threads = omp_get_max_threads();

    int ld = (d + 1 + 7) / 8 * 8;
    double *X = alloc64((size_t)n * d * sizeof(double));
    double *y = alloc64((size_t)n * sizeof(double));
    double *w = alloc64((size_t)d * sizeof(double));
    double *wref = alloc64((size_t)d * sizeof(double));
    double *gbuf = alloc64((size_t)threads * ld * sizeof(double));
    double *times = malloc((size_t)reps * sizeof(double));
    if (!X || !y || !w || !wref || !gbuf || !times) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    memset(gbuf, 0, (size_t)threads * ld * sizeof(double));

    generate(X, y, n, d, seed);

    const char *names[] = {"serial", "omp", "simd", "omp_simd"};
    train_fn fns[] = {train_serial, train_omp, train_simd, train_omp_simd};
    double bref = 0.0, lref = 0.0;
    int all_ok = 1;

    printf("N=%d d=%d K=%d lambda=%g eta=%g threads=%d reps=%d seed=%llu eps=%g tol=%g\n",
           n, d, iters, lam, eta, threads, reps, (unsigned long long)seed, eps, tol);

    for (int v = 0; v < 4; v++) {
        double b = 0.0;
        for (int r = 0; r < reps; r++) {
            memset(w, 0, (size_t)d * sizeof(double));
            b = 0.0;
            double t = omp_get_wtime();
            fns[v](X, y, w, &b, n, d, iters, eta, lam, gbuf, ld);
            times[r] = omp_get_wtime() - t;
        }
        qsort(times, (size_t)reps, sizeof(double), cmp_double);

        double f = loss(X, y, w, b, n, d, lam);
        int zeros = 0;
        for (int j = 0; j < d; j++)
            zeros += fabs(w[j]) < eps;
        if (v == 0) {
            memcpy(wref, w, (size_t)d * sizeof(double));
            bref = b;
            lref = f;
        }
        double dw = 0.0;
        for (int j = 0; j < d; j++)
            if (fabs(w[j] - wref[j]) > dw)
                dw = fabs(w[j] - wref[j]);
        int ok = dw <= tol && fabs(b - bref) <= tol && fabs(f - lref) <= tol * fmax(1.0, fabs(lref));
        all_ok &= ok;
        printf("%-9s time=%.6f s  loss=%.12e  dw=%.3e  zeros=%d/%d  check=%s\n",
               names[v], times[reps / 2], f, dw, zeros, d, ok ? "OK" : "FAIL");
    }

    free(X);
    free(y);
    free(w);
    free(wref);
    free(gbuf);
    free(times);
    return all_ok ? 0 : 2;
}
