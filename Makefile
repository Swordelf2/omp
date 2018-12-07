# C99 extensions are not necessary for OpenMP, but very convenient
CC=gcc
CFLAGS= -g -fopenmp -Wall -std=gnu11
DEFINES=

ifdef N
DEFINES+=-DN=$N
endif

ifdef P_SQRT
DEFINES+=-DP_SQRT=$(P_SQRT)
endif

ifdef TEST
DEFINES+=-DTEST
endif

CFLAGS += $(DEFINES)

main: main.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(OBJS) main
