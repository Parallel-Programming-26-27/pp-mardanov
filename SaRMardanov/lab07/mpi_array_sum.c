#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size, n = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0 && argc == 2) {
        char *end;
        errno = 0;
        long value = strtol(argv[1], &end, 10);
        if (!errno && end != argv[1] && *end == '\0' && value > 0 && value <= INT_MAX)
            n = (int)value;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0 || n % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s N; 0 < N <= INT_MAX, N must be divisible by P\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int count = n / size;
    int *array = rank == 0 ? malloc((size_t)n * sizeof(*array)) : NULL;
    int *local = malloc((size_t)count * sizeof(*local));
    long long *sums = rank == 0 ? malloc((size_t)size * sizeof(*sums)) : NULL;
    int allocated = local != NULL && (rank != 0 || (array != NULL && sums != NULL));
    int all_allocated;
    MPI_Allreduce(&allocated, &all_allocated, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_allocated) {
        if (rank == 0)
            fprintf(stderr, "Allocation failed\n");
        free(array);
        free(local);
        free(sums);
        MPI_Finalize();
        return 1;
    }
    if (rank == 0)
        for (int i = 0; i < n; i++)
            array[i] = i + 1;
    MPI_Scatter(array, count, MPI_INT, local, count, MPI_INT, 0, MPI_COMM_WORLD);
    long long local_sum = 0, gathered = 0, reduced = 0, allreduced = 0;
    for (int i = 0; i < count; i++)
        local_sum += local[i];
    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        for (int i = 0; i < size; i++)
            gathered += sums[i];
    MPI_Reduce(&local_sum, &reduced, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sum, &allreduced, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long long expected = (long long)n * ((long long)n + 1) / 2;
    int ok = 0;
    if (rank == 0)
        ok = gathered == expected && reduced == expected && allreduced == expected;
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("P=%d N=%d expected=%lld Gather=%lld Reduce=%lld Allreduce=%lld check=%s time=%.9f\n",
               size, n, expected, gathered, reduced, allreduced, ok ? "OK" : "FAIL", seconds);
    free(array);
    free(local);
    free(sums);
    MPI_Finalize();
    return ok ? 0 : 2;
}
