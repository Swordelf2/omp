CC=mpicc
CFLAGS= -g -Wall -std=gnu11
DEFINES=

ifndef P
P=4
endif

ifndef N
N=2
endif

ifdef TEST
DEFINES+=-DTEST
endif

CFLAGS += $(DEFINES)

main: main_mpi.c
	$(CC) $(CFLAGS) -o $@ $< -lm

run: main
	mpirun --oversubscribe -n $P ./main $N

clean:
	rm -f $(OBJS) main
