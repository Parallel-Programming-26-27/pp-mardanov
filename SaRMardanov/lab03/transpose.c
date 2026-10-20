#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void transpose_rows(const double *A, double *B, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[(size_t)j * n + i] = A[(size_t)i * n + j];
}

void transpose_cols(const double *A, double *B, int n)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            B[(size_t)j * n + i] = A[(size_t)i * n + j];
}

void transpose_blocked(const double *A, double *B, int n, int b)
{
    for (int ii = 0; ii < n; ii += b)
        for (int jj = 0; jj < n; jj += b) {
            int imax = ii + b < n ? ii + b : n;
            int jmax = jj + b < n ? jj + b : n;
            for (int i = ii; i < imax; i++)
                for (int j = jj; j < jmax; j++)
                    B[(size_t)j * n + i] = A[(size_t)i * n + j];
        }
}

void transpose_rows_omp(const double *A, double *B, int n)
{
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[(size_t)j * n + i] = A[(size_t)i * n + j];
}

void transpose_blocked_omp(const double *A, double *B, int n, int b)
{
#pragma omp parallel for
    for (int ii = 0; ii < n; ii += b)
        for (int jj = 0; jj < n; jj += b) {
            int imax = ii + b < n ? ii + b : n;
            int jmax = jj + b < n ? jj + b : n;
            for (int i = ii; i < imax; i++)
                for (int j = jj; j < jmax; j++)
                    B[(size_t)j * n + i] = A[(size_t)i * n + j];
        }
}

void transpose_inplace(double *A, int n, int b)
{
    for (int ii = 0; ii < n; ii += b)
        for (int jj = ii; jj < n; jj += b) {
            int imax = ii + b < n ? ii + b : n;
            int jmax = jj + b < n ? jj + b : n;
            for (int i = ii; i < imax; i++)
                for (int j = ii == jj ? i + 1 : jj; j < jmax; j++) {
                    double t = A[(size_t)i * n + j];
                    A[(size_t)i * n + j] = A[(size_t)j * n + i];
                    A[(size_t)j * n + i] = t;
                }
        }
}

void transpose_inplace_omp(double *A, int n, int b)
{
#pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += b)
        for (int jj = ii; jj < n; jj += b) {
            int imax = ii + b < n ? ii + b : n;
            int jmax = jj + b < n ? jj + b : n;
            for (int i = ii; i < imax; i++)
                for (int j = ii == jj ? i + 1 : jj; j < jmax; j++) {
                    double t = A[(size_t)i * n + j];
                    A[(size_t)i * n + j] = A[(size_t)j * n + i];
                    A[(size_t)j * n + i] = t;
                }
        }
}

int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char *argv[])
{
    int n = 4000, b = 32, threads = 8;
    const char *version = "blocked_omp";

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) b = atoi(argv[2]);
    if (argc > 3) threads = atoi(argv[3]);
    if (argc > 4) version = argv[4];
    omp_set_num_threads(threads);
    if (getenv("OMP_SCHEDULE") == NULL)
        omp_set_schedule(omp_sched_static, 0);

    size_t len = (size_t)n * n;
    double *A = malloc(len * sizeof(double));
    double *B = malloc(len * sizeof(double));
    double *ref = malloc(len * sizeof(double));
    for (size_t k = 0; k < len; k++)
        A[k] = (double)k;
    transpose_rows(A, ref, n);

    int inplace = strncmp(version, "inplace", 7) == 0;
    double times[5];
    long long bad = 0;
    for (int r = -1; r < 5; r++) {
        if (inplace)
            memcpy(B, A, len * sizeof(double));
        else
            memset(B, 0, len * sizeof(double));

        double t = omp_get_wtime();
        if (strcmp(version, "rows") == 0)
            transpose_rows(A, B, n);
        else if (strcmp(version, "cols") == 0)
            transpose_cols(A, B, n);
        else if (strcmp(version, "blocked") == 0)
            transpose_blocked(A, B, n, b);
        else if (strcmp(version, "rows_omp") == 0)
            transpose_rows_omp(A, B, n);
        else if (strcmp(version, "blocked_omp") == 0)
            transpose_blocked_omp(A, B, n, b);
        else if (strcmp(version, "inplace") == 0)
            transpose_inplace(B, n, b);
        else if (strcmp(version, "inplace_omp") == 0)
            transpose_inplace_omp(B, n, b);
        else {
            fprintf(stderr, "Unknown version: %s\n", version);
            return 1;
        }
        t = omp_get_wtime() - t;

        for (size_t k = 0; k < len; k++)
            if (B[k] != ref[k])
                bad++;
        if (r >= 0)
            times[r] = t;
    }

    qsort(times, 5, sizeof(double), cmp_double);
    double t = times[2];
    double bw = 2.0 * len * sizeof(double) / t / 1e9;

    printf("N=%d b=%d threads=%d version=%s time=%.6f s BW=%.2f GB/s check=%s\n",
           n, b, threads, version, t, bw, bad == 0 ? "OK" : "FAIL");

    free(A);
    free(B);
    free(ref);
    return 0;
}
