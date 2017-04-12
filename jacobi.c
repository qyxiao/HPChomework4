/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "util.h"
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *u, int N, double invhsq)
{
  int i,j;
  double tmp, gres = 0.0, res = 0.0;

  for (i = 1; i <= N; i++){
    for(j=1;j<=N;j++){
    tmp = ((4.0*u[i*(N+2)+j] - u[(i-1)*(N+2)+j] - u[(i+1)*(N+2)+j]-u[i*(N+2)+j-1]-u[i*(N+2)+j+1]) * invhsq - 1);
    res += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&res, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[])
{
  int mpirank, i, j, p, N, lN, iter, max_iters;
 

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);
  int sP = (int)sqrt(p);
  /* compute number of unknowns handled by each process */
  lN = N / sP;


  /*
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  */
   

  MPI_Status status[lN];
  MPI_Request request_out1[lN], request_in1[lN],request_out2[lN], request_in2[lN];
  MPI_Request request_out3[lN], request_in3[lN],request_out4[lN], request_in4[lN];

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  timestamp_type time1, time2;
  get_timestamp(&time1);

  /* Allocation of vectors, including left and right ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
    /* interleaf computation and communication: compute the first
     * and last value, which are communicated with non-blocking
     * send/recv. During that communication, do all the local work */

    /* Jacobi step for the left and right most points */
    for(j=1;j<=lN;j++){
      lunew[1*(lN+2)+j] = 0.25 * (hsq + lu[j] + lu[2*(lN+2)+j] + lu[(lN+2)+j-1] + lu[(lN+2)+j+1]);
      lunew[lN*(lN+2)+j] = 0.25 * (hsq + lu[(lN-1)*(lN+2)+j] + lu[(lN+1)*(lN+2)+j] + lu[lN*(lN+2)+j-1] + lu[lN*(lN+2)+j+1]);
      lunew[j*(lN+2)+1] = 0.25 * (hsq + lu[(j-1)*(lN+2)+1] + lu[(j+1)*(lN+2)+1] + lu[j*(lN+2)] + lu[j*(lN+2)+2]);
      lunew[j*(lN+2)+lN] = 0.25 * (hsq + lu[(j-1)*(lN+2)+lN] + lu[(j+1)*(lN+2)+lN] + lu[j*(lN+2)+lN-1] + lu[j*(lN+2)+lN+1]);
    }
    

    if(mpirank-sP>=0){
      for(i=1;i<=lN;i++){
      MPI_Irecv(&(lunew[i]), 1, MPI_DOUBLE, mpirank-sP, i, MPI_COMM_WORLD, &request_in1[i-1]);
      MPI_Isend(&(lunew[lN+2+i]), 1, MPI_DOUBLE, mpirank-sP, i, MPI_COMM_WORLD, &request_out1[i-1]);
    }
    }

    if(mpirank+sP<p){
      for(i=1;i<=lN;i++){
      MPI_Irecv(&(lunew[(lN+1)*(lN+2)+i]), 1, MPI_DOUBLE, mpirank+sP, i, MPI_COMM_WORLD, &request_in2[i-1]);
      MPI_Isend(&(lunew[(lN)*(lN+2)+i]), 1, MPI_DOUBLE, mpirank+sP, i, MPI_COMM_WORLD, &request_out2[i-1]);
    }
    }

    if(mpirank % sP !=0){
      for(i=1;i<=lN;i++){
      MPI_Irecv(&(lunew[i*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, i, MPI_COMM_WORLD, &request_in3[i-1]);
      MPI_Isend(&(lunew[i*(lN+2)+1]), 1, MPI_DOUBLE, mpirank-1, i, MPI_COMM_WORLD, &request_out3[i-1]);
    }
    }


    if((mpirank+1) % sP !=0){
      for(i=1;i<=lN;i++){
      MPI_Irecv(&(lunew[i*(lN+2)+lN+1]), 1, MPI_DOUBLE, mpirank+1, i, MPI_COMM_WORLD, &request_in4[i-1]);
      MPI_Isend(&(lunew[i*(lN+2)+lN]), 1, MPI_DOUBLE, mpirank+1, i, MPI_COMM_WORLD, &request_out4[i-1]);
    }
    }


    /* Jacobi step for all the inner points */
    for (i = 2; i < lN; i++){
      for(j = 2; j < lN; j++){
        lunew[i*(lN+2)+j] = 0.25 * (hsq + lu[(i-1)*(lN+2)+j] + lu[(i+1)*(lN+2)+j] + lu[i*(lN+2)+j-1] + lu[i*(lN+2)+j+1]);
      }
    }

    /* check if Isend/Irecv are done */  
    if(mpirank-sP>=0) {
      MPI_Waitall(lN, request_out1, status);
      MPI_Waitall(lN, request_in1, status);
    }
    if(mpirank+sP<p) {
      MPI_Waitall(lN, request_out2, status);
      MPI_Waitall(lN, request_in2, status);
    }
    if(mpirank % sP !=0) {
      MPI_Waitall(lN, request_out3, status);
      MPI_Waitall(lN, request_in3, status);
    }
    if((mpirank+1) % sP !=0) {
      MPI_Waitall(lN, request_out4, status);
      MPI_Waitall(lN, request_in4, status);
    }
    

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  get_timestamp(&time2);
  double elapsed = timestamp_diff_in_seconds(time1,time2);
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}