CC=mpicc 

EXECS= mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 jacobi ssort

all: ${EXECS}

mpi1: mpi_solved1.c
	${CC}  $^ -o mpi_solved1

mpi2: mpi_solved2.c
	${CC}  $^ -o mpi_solved2

mpi3: mpi_solved3.c
	${CC}  $^ -o mpi_solved3

mpi4: mpi_solved4.c
	${CC}  $^ -o mpi_solved4

mpi5: mpi_solved5.c
	${CC}  $^ -o mpi_solved5

mpi6: mpi_solved6.c
	${CC}  $^ -o mpi_solved6

mpi7: mpi_solved7.c
	${CC}  $^ -o mpi_solved7

jacobi: jacobi-mpi2D.c
	${CC}  $^ -o jacobi-mpi2D

ssort: ssort.c
	${CC}  $^ -o ssort

clean:
	rm -f ${EXECS}