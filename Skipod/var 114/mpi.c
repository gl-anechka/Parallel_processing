#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define Min(a,b) ((a)<(b)?(a):(b))

#define N (2*2*2*2*2*2*2*2*2*2*2*2+2)

double maxeps = 0.1e-7;
int itmax     = 10;
double eps;
double **A;

static double **alloc_2d(int rows, int cols) {
    double *data = (double*)calloc((size_t)rows * (size_t)cols, sizeof(double));
    if (!data) {
        fprintf(stderr, "alloc_2d: cannot allocate data\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double **a = (double**)malloc((size_t)rows * sizeof(double*));
    if (!a) {
        fprintf(stderr, "alloc_2d: cannot allocate row pointers\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < rows; ++i)
        a[i] = data + (size_t)i * (size_t)cols;
    return a;
}

static void free_2d(double **a) {
    if (!a) return;
    free(a[0]);
    free(a);
}

void init(int rank, int size, int istart, int iend) {
    int rows = iend - istart + 1;
    for (int il = 0; il < rows + 2; ++il) {
        int gi;
        if (il == 0)
            gi = (rank == 0) ? 0 : istart - 1;
        else if (il == rows + 1)
            gi = (rank == size - 1) ? N - 1 : iend + 1;
        else
            gi = istart + (il - 1);

        for (int j = 0; j < N; ++j) {
            if (gi == 0 || gi == N - 1 || j == 0 || j == N - 1)
                A[il][j] = 0.0;
            else
                A[il][j] = 2.0 + gi + j;
        }
    }
}

void relax(int rank, int size, int istart, int iend) {
    int rows = iend - istart + 1;
    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    MPI_Status st;

    if (up != MPI_PROC_NULL) {
        MPI_Sendrecv(A[1], N, MPI_DOUBLE, up,   0,
                     A[0], N, MPI_DOUBLE, up,   1,
                     MPI_COMM_WORLD, &st);
    }
    if (down != MPI_PROC_NULL) {
        MPI_Sendrecv(A[rows],     N, MPI_DOUBLE, down, 1,
                     A[rows + 1], N, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, &st);
    }

    double eps_local = 0.0;
    int gi_start = Max(istart, 1);
    int gi_end   = Min(iend,   N - 2);

    for (int gi = gi_start; gi <= gi_end; ++gi) {
        int il = gi - istart + 1;
        for (int j = 1; j <= N - 2; ++j) {
            double e = A[il][j];
            A[il][j] = (2.0 * A[il-1][j] + A[il+1][j] +
                        2.0 * A[il][j-1] + A[il][j+1]) / 6.0;
            double d = fabs(e - A[il][j]);
            eps_local = Max(eps_local, d);
        }
    }

    eps = eps_local;
}

void verify(int rank, int size, int istart, int iend) {
    int rows = iend - istart + 1;
    double s_local = 0.0;
    double coef = 1.0 / ((double)N * (double)N);

    for (int gi = istart; gi <= iend; ++gi) {
        int il = gi - istart + 1;
        for (int j = 0; j < N; ++j)
            s_local += A[il][j] * (gi + 1) * (j + 1) * coef;
    }

    double s = 0.0;
    MPI_Reduce(&s_local, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("   S = %f\n", s);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int inner = N - 2;
    int base = inner / size;
    int rem  = inner % size;

    int local_rows, istart;
    if (rank < rem) {
        local_rows = base + 1;
        istart = 1 + rank * local_rows;
    } else {
        local_rows = base;
        istart = 1 + rem * (base + 1) + (rank - rem) * base;
    }
    int iend = istart + local_rows - 1;

    A = alloc_2d(local_rows + 2, N);

    double t0 = MPI_Wtime();

    init(rank, size, istart, iend);

    for (int it = 1; it <= itmax; ++it) {
        eps = 0.0;
        relax(rank, size, istart, iend);
        double eps_global = 0.0;
        MPI_Allreduce(&eps, &eps_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        eps = eps_global;
        if (rank == 0)
            printf("it = %4d eps = %e\n", it, eps);
        if (eps < maxeps) break;
    }

    verify(rank, size, istart, iend);

    double t1 = MPI_Wtime();
    if (rank == 0)
        printf("Execution time (MPI): %f\n", t1 - t0);

    free_2d(A);
    MPI_Finalize();
    return 0;
}
