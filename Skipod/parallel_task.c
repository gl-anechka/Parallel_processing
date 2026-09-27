#include <omp.h>
#include "params.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

double maxeps = 0.1e-7;
int itmax = 100;
double eps;
double A [N][N][N] __attribute__((aligned(64)));
double B [N][N][N] __attribute__((aligned(64)));

void relax();
void resid();
void init();
void verify();

int main(int an, char **as)
{
	printf("===== Parallel program on tasks =====\n\n");
	double start = omp_get_wtime();

    int flag = 0;

    #pragma omp parallel
    {
		printf("parallel ");
		#pragma omp master
		{
            init();
			for(int it=1; it<=itmax; it++)
			{
				eps = 0.;
				relax();
				resid();
				printf( "it=%4i   eps=%f\n", it, eps);
				if (eps < maxeps) break;
			}
			verify();
		}
    }

	double end = omp_get_wtime();
	double result_time = (end - start) * 1000;
	printf("\nExecution time: %lf ms\n", result_time);

	return 0;
}

void init()
{
	for(int i=0; i<=N-1; i+=BLOCK_SIZE)
	for(int j=0; j<=N-1; j+=BLOCK_SIZE)
	for(int k=0; k<=N-1; k+=BLOCK_SIZE)
	{
		#pragma omp task firstprivate(i,j,k)
		for (int ii = i; ii < Min(i+BLOCK_SIZE, N); ii++)
		for (int jj = j; jj < Min(j+BLOCK_SIZE, N); jj++)
		for (int kk = k; kk < Min(k+BLOCK_SIZE, N); kk++)
		{
			if(ii==0 || ii==N-1 || jj==0 || jj==N-1 || kk==0 || kk==N-1) A[ii][jj][kk]= 0.;
			else A[ii][jj][kk]= (4. + ii + jj + kk);
		}
	}
	#pragma omp taskwait
}

void relax()
{
	for(int i=2; i<=N-3; i+=BLOCK_SIZE)
	for(int j=2; j<=N-3; j+=BLOCK_SIZE)
	for(int k=2; k<=N-3; k+=BLOCK_SIZE)
	{
		#pragma omp task firstprivate(i,j,k)
		for (int ii = i; ii < Min(i+BLOCK_SIZE, N-2); ii++)
		for (int jj = j; jj < Min(j+BLOCK_SIZE, N-2); jj++)
		for (int kk = k; kk < Min(k+BLOCK_SIZE, N-2); kk++)
		{
			B[ii][jj][kk]=(A[ii-1][jj][kk]+A[ii+1][jj][kk]+A[ii][jj-1][kk]+A[ii][jj+1][kk]+A[ii][jj][kk-1]+A[ii][jj][kk+1]+
				A[ii-2][jj][kk]+A[ii+2][jj][kk]+A[ii][jj-2][kk]+A[ii][jj+2][kk]+A[ii][jj][kk-2]+A[ii][jj][kk+2])/12.;
		}
	}
	#pragma omp taskwait
}

void resid()
{
	int t_count = omp_get_max_threads();
	double eps_loc[t_count];
	for (int i = 0; i < t_count; i++) eps_loc[i] = 0.0;

	for(int i=1; i<=N-2; i+=BLOCK_SIZE)
	for(int j=1; j<=N-2; j+=BLOCK_SIZE)
	for(int k=1; k<=N-2; k+=BLOCK_SIZE)
	{
		#pragma omp task firstprivate(i,j,k) shared(eps_loc)
		{
			int tid = omp_get_thread_num();
			double e_loc = 0.0;

			for (int ii = i; ii < Min(i+BLOCK_SIZE, N-1); ii++)
			for (int jj = j; jj < Min(j+BLOCK_SIZE, N-1); jj++)
			for (int kk = k; kk < Min(k+BLOCK_SIZE, N-1); kk++)
			{
				double e = fabs(A[ii][jj][kk] - B[ii][jj][kk]);         
				A[ii][jj][kk] = B[ii][jj][kk]; 
				e_loc = Max(e_loc, e);
			}

			eps_loc[tid] = Max(e_loc, eps_loc[tid]);
		}
	}
	#pragma omp taskwait

	for (int i = 0; i < t_count; i++)
		if (eps_loc[i] > eps) eps = eps_loc[i];
}

void verify()
{
	double s = 0.;
	int t_count = omp_get_max_threads();
	double s_loc[t_count];
	for (int i = 0; i < t_count; i++) s_loc[i] = 0.0;

	for(int i=0; i<=N-1; i+=BLOCK_SIZE)
	for(int j=0; j<=N-1; j+=BLOCK_SIZE)
	for(int k=0; k<=N-1; k+=BLOCK_SIZE)
	{
		#pragma omp task firstprivate(i,j,k) shared(s_loc)
		{
			int tid = omp_get_thread_num();
			double temp = 0.0;

			for (int ii = i; ii < Min(i+BLOCK_SIZE, N); ii++)
			for (int jj = j; jj < Min(j+BLOCK_SIZE, N); jj++)
			for (int kk = k; kk < Min(k+BLOCK_SIZE, N); kk++)
			{
				temp += A[ii][jj][kk]*(ii+1)*(jj+1)*(kk+1)/(N*N*N);
			}

			s_loc[tid] += temp;
		}
	}
	#pragma omp taskwait

	for (int i = 0; i < t_count; i++)
		s += s_loc[i];
	printf("  S = %f\n",s);
}
