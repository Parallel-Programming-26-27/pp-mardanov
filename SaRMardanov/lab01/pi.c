#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double pi_sequential(long long n) {
    double sum = 0.0;
    for (long long i = 0; i < n; i++) {
        double sign = (i & 1) ? -1.0 : 1.0;
        sum += sign / (2.0 * (double)i + 1.0);
    }
    return 4.0 * sum;
}

static double pi_parallel(long long n) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        double sign = (i & 1) ? -1.0 : 1.0;
        sum += sign / (2.0 * (double)i + 1.0);
    }
    return 4.0 * sum;
}

static void print_result(const char *label, double pi, double time, int threads) {
    printf("%s\n", label);
    printf("  pi        = %.15f\n", pi);
    printf("  error     = %.3e\n", fabs(pi - M_PI));
    printf("  time      = %.4f s\n", time);
    printf("  threads   = %d\n", threads);
}

int main(int argc, char *argv[]) {
    long long n = 1000000000LL;
    int threads = omp_get_max_threads();

    if (argc > 1) n = (long long)strtod(argv[1], NULL);
    if (argc > 2) threads = atoi(argv[2]);
    if (n <= 0 || threads <= 0) {
        fprintf(stderr, "Usage: %s [n] [threads]\n", argv[0]);
        return 1;
    }
    omp_set_num_threads(threads);

    printf("n = %lld\n\n", n);

    double t0 = omp_get_wtime();
    double pi_seq = pi_sequential(n);
    double t_seq = omp_get_wtime() - t0;
    print_result("Sequential:", pi_seq, t_seq, 1);

    t0 = omp_get_wtime();
    double pi_par = pi_parallel(n);
    double t_par = omp_get_wtime() - t0;
    print_result("\nOpenMP:", pi_par, t_par, threads);

    printf("\nSpeedup   = %.2f\n", t_seq / t_par);
    return 0;
}
