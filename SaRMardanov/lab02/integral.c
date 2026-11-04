#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double f(double x)
{
    return exp(x) * sin(x);
}

double integral_seq(double a, double b, long long n)
{
    double h = (b - a) / n;
    double sum = 0;
    for (long long i = 0; i < n; i++)
        sum += f(a + (i + 0.5) * h);
    return sum * h;
}

double integral_par(double a, double b, long long n, int threads)
{
    double h = (b - a) / n;
    double sum = 0;
#pragma omp parallel for num_threads(threads) reduction(+:sum)
    for (long long i = 0; i < n; i++)
        sum += f(a + (i + 0.5) * h);
    return sum * h;
}

int main(int argc, char *argv[])
{
    double a = 0, b = M_PI;
    long long n = 100000000;
    int threads = 8;

    if (argc > 1) a = atof(argv[1]);
    if (argc > 2) b = atof(argv[2]);
    if (argc > 3) n = atoll(argv[3]);
    if (argc > 4) threads = atoi(argv[4]);

    double t, res;

    if (argc > 5) {
        double eps = atof(argv[5]);
        long long k;
        double prev, cur;

        k = n;
        t = omp_get_wtime();
        prev = integral_seq(a, b, k);
        while (1) {
            k *= 2;
            cur = integral_seq(a, b, k);
            if (fabs(cur - prev) < eps) break;
            prev = cur;
        }
        t = omp_get_wtime() - t;
        printf("Последовательно (потоков: 1): n = %lld, I = %.12f, время = %.6f с\n", k, cur, t);

        k = n;
        t = omp_get_wtime();
        prev = integral_par(a, b, k, threads);
        while (1) {
            k *= 2;
            cur = integral_par(a, b, k, threads);
            if (fabs(cur - prev) < eps) break;
            prev = cur;
        }
        t = omp_get_wtime() - t;
        printf("Параллельно (потоков: %d): n = %lld, I = %.12f, время = %.6f с\n", threads, k, cur, t);
        return 0;
    }

    t = omp_get_wtime();
    res = integral_seq(a, b, n);
    t = omp_get_wtime() - t;
    printf("Последовательно (потоков: 1): I = %.12f, время = %.6f с\n", res, t);

    t = omp_get_wtime();
    res = integral_par(a, b, n, threads);
    t = omp_get_wtime() - t;
    printf("Параллельно (потоков: %d): I = %.12f, время = %.6f с\n", threads, res, t);

    return 0;
}
