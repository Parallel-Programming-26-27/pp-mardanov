#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*matmul_fn)(const double *A, const double *B, double *C, int n, int bs);

static void ijk_seq(const double *A, const double *B, double *C, int n, int bs)
{
    (void)bs;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
}

static void ijk_omp(const double *A, const double *B, double *C, int n, int bs)
{
    (void)bs;
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
}

static void ikj_seq(const double *A, const double *B, double *C, int n, int bs)
{
    (void)bs;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++)
                C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
}

static void ikj_omp(const double *A, const double *B, double *C, int n, int bs)
{
    (void)bs;
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++)
                C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
}

static void blocked_omp(const double *A, const double *B, double *C, int n, int bs)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += bs)
        for (int jj = 0; jj < n; jj += bs) {
            int imax = ii + bs < n ? ii + bs : n;
            int jmax = jj + bs < n ? jj + bs : n;
            for (int kk = 0; kk < n; kk += bs) {
                int kmax = kk + bs < n ? kk + bs : n;
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        double a = A[(size_t)i * n + k];
                        for (int j = jj; j < jmax; j++)
                            C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
                    }
            }
        }
}

static void fill(double *M, size_t len, unsigned int seed)
{
    unsigned int x = seed;
    for (size_t i = 0; i < len; i++) {
        x = x * 1103515245u + 12345u;
        M[i] = (double)((x >> 16) & 0x7fff) / 32767.0 * 2.0 - 1.0;
    }
}

static double max_rel_error(const double *C, const double *ref, size_t len)
{
    double err = 0.0;
    for (size_t i = 0; i < len; i++) {
        double d = fabs(C[i] - ref[i]) / (fabs(ref[i]) > 1.0 ? fabs(ref[i]) : 1.0);
        if (d > err)
            err = d;
    }
    return err;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char *argv[])
{
    int n = 512, bs = 32, reps = 5;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) bs = atoi(argv[2]);
    if (argc > 3) reps = atoi(argv[3]);
    if (n <= 0 || bs <= 0 || reps <= 0) {
        fprintf(stderr, "Usage: OMP_NUM_THREADS=<p> %s [N] [block] [reps]\n", argv[0]);
        return 1;
    }

    size_t len = (size_t)n * n;
    double *A = malloc(len * sizeof(double));
    double *B = malloc(len * sizeof(double));
    double *C = malloc(len * sizeof(double));
    double *ref = malloc(len * sizeof(double));
    double *times = malloc((size_t)reps * sizeof(double));
    if (!A || !B || !C || !ref || !times) {
        fprintf(stderr, "Allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(ref);
        free(times);
        return 1;
    }

    fill(A, len, 1u);
    fill(B, len, 2u);

    const char *names[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};
    matmul_fn fns[] = {ijk_seq, ijk_omp, ikj_seq, ikj_omp, blocked_omp};
    const double tol = 1e-9;
    int all_ok = 1;

    printf("N=%d block=%d threads=%d reps=%d\n", n, bs, omp_get_max_threads(), reps);

    for (int v = 0; v < 5; v++) {
        double err = 0.0;
        for (int r = 0; r < reps; r++) {
            memset(C, 0, len * sizeof(double));
            double t = omp_get_wtime();
            fns[v](A, B, C, n, bs);
            times[r] = omp_get_wtime() - t;
            if (v == 0 && r == 0)
                memcpy(ref, C, len * sizeof(double));
            double e = max_rel_error(C, ref, len);
            if (e > err)
                err = e;
        }
        qsort(times, (size_t)reps, sizeof(double), cmp_double);
        int ok = err <= tol;
        all_ok &= ok;
        printf("%-12s time=%.6f s  max_err=%.3e  check=%s\n",
               names[v], times[reps / 2], err, ok ? "OK" : "FAIL");
    }

    free(A);
    free(B);
    free(C);
    free(ref);
    free(times);
    return all_ok ? 0 : 2;
}
