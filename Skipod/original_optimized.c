#include <omp.h>
#include "params.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

double maxeps = 0.1e-7;
int itmax = 100;
double eps;
double A [N][N][N], B [N][N][N];

void relax();
void resid();
void init();
void verify();

int main(int an, char **as)
{
	printf("===== Original optimized program =====\n\n");
	double start = omp_get_wtime();

	int it;
	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		printf( "it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps) break;
	}
	verify();

	double end = omp_get_wtime();
	double result_time = (end - start) * 1000;
	printf("\nExecution time: %lf ms\n", result_time);

	return 0;
}

void init()
{ 
	for(int i=0; i<=N-1; i++)
	for(int j=0; j<=N-1; j++)
	for(int k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[i][j][k]= 0.;
		else A[i][j][k]= (4. + i + j + k);
	}
}

void relax()
{
	for(int i=2; i<=N-3; i++)
	for(int j=2; j<=N-3; j++)
	for(int k=2; k<=N-3; k++)
	{
		B[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1]+
			A[i-2][j][k]+A[i+2][j][k]+A[i][j-2][k]+A[i][j+2][k]+A[i][j][k-2]+A[i][j][k+2])/12.;
	}
}

void resid()
{
	for(int i=1; i<=N-2; i++)
	for(int j=1; j<=N-2; j++)
	for(int k=1; k<=N-2; k++)
	{
		double e;
		e = fabs(A[i][j][k] - B[i][j][k]);         
		A[i][j][k] = B[i][j][k]; 
		eps = Max(eps,e);
	}
}

void verify()
{
	double s = 0.;
	for(int i=0; i<=N-1; i++)
	for(int j=0; j<=N-1; j++)
	for(int k=0; k<=N-1; k++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	printf("  S = %f\n",s);
}
