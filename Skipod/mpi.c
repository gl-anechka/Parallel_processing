#include <mpi.h>
#include "params.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define A(i,j,k) A[((i)+2)*N*N + (j)*N + (k)]
#define B(i,j,k) B[((i)+2)*N*N + (j)*N + (k)]

double maxeps = 0.1e-7;
int itmax = 100;
double eps;
int proc, proc_num;
int startslice, endslice, nslice;
double *A, *B;

MPI_Request recv[2], send[2];
MPI_Status stat[2];

void relax();
void resid();
void init();
void verify();

int main(int an, char **as)
{
	int it;

	MPI_Init(&an, &as);
	MPI_Comm_size(MPI_COMM_WORLD, &proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);
	MPI_Barrier(MPI_COMM_WORLD);

	if (proc_num == 0) printf("===== MPI program =====\n\n");
	double start_time = MPI_Wtime();

	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		if (proc_num == 0) printf( "it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps) break;
	}
	verify();

	MPI_Barrier(MPI_COMM_WORLD);

	double end_time = MPI_Wtime();
	double result_time = (end_time - start_time) * 1000;
	if (proc_num == 0) printf("\nExecution time: %lf ms\n", result_time);

	free(A);
	free(B);
	MPI_Finalize();
	return 0;
}

void init()
{ 
	startslice = round(((double)N / proc) * proc_num);
	endslice = round(((double)N / proc) * (proc_num + 1));
	nslice = endslice - startslice;

	A = calloc(N * N * (nslice + 4), sizeof(double));
	B = calloc(N * N * (nslice + 4), sizeof(double));

	for(int i=0; i<nslice; i++)
	{
		int global_i = i + startslice;
		for(int j=1; j<N-1; j++)
		for(int k=1; k<N-1; k++)
		{
			if (global_i != N-1 && global_i != 0)
				A(i,j,k) = (4. + global_i + j + k);
		}
	}
}

void relax()
{
	if (proc > 1) {
		if (proc_num != 0) {
			MPI_Irecv(&A(-2,0,0), N*N*2, MPI_DOUBLE, proc_num-1, 0, MPI_COMM_WORLD, &recv[0]);
		}
		if (proc_num != proc - 1) {
			MPI_Isend(&A(nslice-2,0,0), N*N*2, MPI_DOUBLE, proc_num+1, 0, MPI_COMM_WORLD, &send[1]);
		}
		if (proc_num != proc - 1) {
			MPI_Irecv(&A(nslice,0,0), N*N*2, MPI_DOUBLE, proc_num+1, 1, MPI_COMM_WORLD, &recv[1]);
		}
		if (proc_num != 0) {
			MPI_Isend(&A(0,0,0), N*N*2, MPI_DOUBLE, proc_num-1, 1, MPI_COMM_WORLD, &send[0]);
		}
	

		if (proc_num == 0) {
			MPI_Wait(&recv[1], &stat[0]);
		} else if (proc_num == proc - 1) {
			MPI_Wait(&recv[0], &stat[1]);
		} else {
			MPI_Waitall(2, recv, stat);
		}
	}

	int local_start, local_end;

	if (proc_num == 0) local_start = 2;
	else local_start = 0;

	if (proc_num == proc - 1) local_end = nslice - 2;
	else local_end = nslice;

	for(int i=local_start; i<local_end; i++)
	for(int j=2; j<=N-3; j++)
	for(int k=2; k<=N-3; k++)
	{
		B(i,j,k) = (A(i-1,j,k)+A(i+1,j,k)+A(i,j-1,k)+A(i,j+1,k)+A(i,j,k-1)+A(i,j,k+1)+
			A(i-2,j,k)+A(i+2,j,k)+A(i,j-2,k)+A(i,j+2,k)+A(i,j,k-2)+A(i,j,k+2))/12.;
	}

	if (proc > 1) {
		if (proc_num == 0) {
			MPI_Wait(&send[1], &stat[0]);
		} else if (proc_num == proc - 1) {
			MPI_Wait(&send[0], &stat[1]);
		} else {
			MPI_Waitall(2, send, stat);
		}
	}
}

void resid()
{
	int local_start, local_end;
	
	if (proc_num == 0) local_start = 1;
	else local_start = 0;

	if (proc_num == proc - 1) local_end = nslice - 1;
	else local_end = nslice;

	double local_eps = 0.;

	for(int i=local_start; i<local_end; i++)
	for(int j=1; j<=N-2; j++)
	for(int k=1; k<=N-2; k++)
	{
		double e;
		e = fabs(A(i,j,k) - B(i,j,k));         
		A(i,j,k) = B(i,j,k); 
		local_eps = Max(local_eps, e);
	}

	MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void verify()
{
	double s = 0.;
	for(int i=0; i<nslice; i++)
	{
		int global_i = i + startslice;

		for(int j=0; j<=N-1; j++)
		for(int k=0; k<=N-1; k++)
		{
			s += A(i,j,k)*(global_i+1)*(j+1)*(k+1)/(double)(N*N*N);
		}
	}

	double global_s = 0.;
	MPI_Allreduce(&s, &global_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if (proc_num == 0) printf("  S = %f\n", global_s);
}