#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define Max(a,b) ((a)>(b)?(a):(b))

#define N (2*2*2*2*2*2*2*2+2)
double maxeps = 0.1e-7;
int itmax = 1000;
double eps;
double A[N][N];

void relax(void);
void init(void);
void verify(void);

int main(int argc, char **argv)
{
    int it;
    double start_time = omp_get_wtime();

    init();

    for (it = 1; it <= itmax; ++it) {
        eps = 0.0;
        relax();
        printf("it = %4d eps = %f\n", it, eps);
        if (eps < maxeps) break;
    }

    verify();
    printf("Execution time: %f seconds\n", omp_get_wtime() - start_time);
    return 0;
}

void init(void)
{
    int i, j;
    for (j = 0; j <= N-1; ++j)
    for (i = 0; i <= N-1; ++i) {
        if (i == 0 || i == N -1 || j == 0 || j == N -1)
            A[i][j] = 0.0;
        else
            A[i][j] = 2.0 + i + j;
    }
}

void relax(void)
{
    int i, j;
    for (j = 1; j <= N-2; ++j)
    for (i = 1; i <= N-2; ++i) {
        double e = A[i][j];
        A[i][j] = (2.0 * A[i-1][j] + A[i+1][j] + 2.0 * A[i][j-1] + A[i][j+1]) / 6.0;
        eps = Max(eps, fabs (e - A[i][j]));
    }
}

void verify(void)
{
    int i, j;
    double s = 0.0;

    for (j = 0; j <= N-1; ++j)
    for (i = 0; i <= N-1; ++i)
    s += A[i][j] * (i+1) * (j+1) / (double)(N*N);

    printf ("   S = %f\n", s);
}