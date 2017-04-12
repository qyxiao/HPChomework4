/* Parallel sample sort
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>


static int compare(const void *a, const void *b)
{
  int *da = (int *)a;
  int *db = (int *)b;

  if (*da > *db)
    return 1;
  else if (*da < *db)
    return -1;
  else
    return 0;
}

int main( int argc, char *argv[])
{
  int rank, size, gap=5;
  int i, N, subsum, index;
  int *vec, *svec, *rvec, *spliter, *counter, *Rcounter;
  int *receiver;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  /* Number of random numbers per processor (this should be increased
   * for actual tests or could be passed in through the command line */
  N = 100;
  
  vec = calloc(N, sizeof(int));
  /* seed random number generator differently on every core */
  srand((unsigned int) (rank + 393919));

  /* fill vector with random integers */
  for (i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);
  qsort(vec, N, sizeof(int), compare);
    
  svec = calloc(gap, sizeof(int));
  for(i=1;i<=gap;i++){
      svec[i-1]=vec[(N/gap)*i];
  }

  if(rank==0){
    rvec = calloc(size*gap, sizeof(int));
  }

  MPI_Gather(svec,gap,MPI_INT,rvec,size*gap,MPI_INT,0,MPI_COMM_WORLD);
  spliter = calloc(size-1, sizeof(int));

  if(rank==0){
    qsort(rvec, gap*size, sizeof(int), compare);  
    for(i=0,i<size-1,i++){
      spliter[i]=rvec[(i+1)*gap];
    }
  }
  
  MPI_Bcast(spliter,size-1,MPI_INT,0,MPI_COMM_WORLD);
  counter = calloc(size, sizeof(int));
  Rcounter = calloc(size, sizeof(int));
  subsum = 0;
  index = 0;
  for(i=0;i<N;i++){
    if(vec[i]>spliter[index]){
      counter[index]=subsum;
      subsum=0;
      index++;
    }else{
      subsum++;
    }
  }
  if(index<N){
    counter[index]=subsum;
  }


  MPI_Alltoall(counter,1,MPI_INT,Rcounter,1,MPI_INT,MPI_COMM_WORLD);
  
  subsum=0;
  for(i=0;i<size;i++){
    subsum = subsum + Rcounter[i];
  }
  

  MPI_Status status_out[N], status_in[subsum];
  MPI_Request request_out[N], request_in[subsum];

  receiver = calloc(subsum, sizeof(int));
  index=0;
  for(i=0;i<size;i++){
    while(counter[i]>0){
      MPI_Isend(&(vec[index]),1,MPI_INT,i,counter[i],MPI_COMM_WORLD, &request_out[index]);
      index++;
      counter[i]--;
    }
  }

  index=0;
  for(i=0;i<size;i++){
    while(Rcounter[i]>0){
      MPI_Irecv(&(receiver[index]), 1, MPI_INT, i, Rcounter[i], MPI_COMM_WORLD, &request_in[index]);
      index++;
      Rcounter[i]--;
    }
  }
  
  
  MPI_Waitall(subsum, request_in, status_in);
  qsort(receiver, subsum, sizeof(int), compare);
  {
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%d.txt", rank);
    fd = fopen(filename,"w+");
    for(i = 0; i < subsum; i++)
      fprintf(fd, "  %d\n", receiver[i]);

    fclose(fd);
    printf("Array received at the end is stored in output%d.txt\n", rank);
  }
  MPI_Waitall(N, request_out, status_out);
  /* do a local sort */

  /* every processor writes its result to a file */

  free(vec);free(svec);free(rvec);free(spliter);free(counter);
  free(Rcounter);free(receiver);free(request_out);free(request_in);
  free(status_out);free(status_in);

  MPI_Finalize();
  return 0;
}
