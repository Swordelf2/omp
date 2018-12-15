#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/**
  * Computing C = A * B
  * A, B, C here are corresponding submatrices of each process
  * Note that computing submatrix C[i][j] is assigned to the process with
  * procid == i * P_SQRT + j
  */

#ifndef RAND_ABS_MAX
#define RAND_ABS_MAX 2.0
#endif

// Returns a random value in [-RAND_ABS_MAX; RAND_ABS_MAX] 
static inline double
get_rand()
{
    return ((double) rand() / RAND_MAX) * (RAND_ABS_MAX * 2) - RAND_ABS_MAX;
}

void
multiply_matrices(double *A, double *B, double *C, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

#ifdef TEST
// convers from Awhole[P_SQRT][P_SQRT][N_SUB][N_SUB] to Aplain[N][N]
void
blocks_to_plain(double *Awhole, double *Aplain, size_t N, size_t N_SUB, size_t P_SQRT)
{
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t i0 = (i / N_SUB) * P_SQRT * N_SUB * N_SUB;
            size_t i1 = (j / N_SUB) * N_SUB * N_SUB;
            size_t i2 = (i % N_SUB) * N_SUB;
            size_t i3 = j % N_SUB;
            Aplain[i * N + j] = Awhole[i0 + i1 + i2 + i3];
        }
    }
}

void
print_matrix(double *A, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            printf("%f ", A[i * N + j]);
        }
        putchar('\n');
    }
}
#endif

enum Args
{
    ARG_N = 1
};

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOFBF, 0);

    /* Initialization */

    // size of matrices
    size_t N = strtoul(argv[ARG_N], NULL, 0);
    // number of processes
    size_t P;
    size_t P_SQRT;
    size_t N_SUB;

    double *A, *B, *C, *Atemp, *Btemp;
    double *Awhole, *Bwhole, *Cwhole;
    MPI_Status status;
    MPI_Init(&argc, &argv);

    int procid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    P = numprocs;
    P_SQRT = lround(sqrt(P));
    if (P_SQRT * P_SQRT != P) {
        if (procid == 0) {
            fprintf(stderr, "P is not a perfect square\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (N % P_SQRT) {
        if (procid == 0) {
            fprintf(stderr, "N is not a multiple of P_SQRT\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    N_SUB = N / P_SQRT;

    A = malloc(N_SUB * N_SUB * sizeof(*A));
    Atemp = malloc(N_SUB * N_SUB * sizeof(*Atemp));
    B = malloc(N_SUB * N_SUB * sizeof(*B));
    Btemp = malloc(N_SUB * N_SUB * sizeof(*Btemp));
    C = malloc(N_SUB * N_SUB * sizeof(*C));

    if (procid == 0) {
        printf("P_SQRT = %zu\n", P_SQRT);
        printf("P = %zu\n", P);
        printf("N = %zu\n", N);
        printf("N_SUB = %zu\n", N_SUB);
        fflush(stdout);
    }

    const size_t n1 = P_SQRT * N_SUB * N_SUB;
    const size_t n2 = N_SUB * N_SUB;
    const size_t subsize = N_SUB * N_SUB;
    if (procid == 0) {
        // Note that these are not A[N][N] but A[P_SQRT][P_SQRT][N_SUB][N_SUB]
        Awhole = malloc(N * N * sizeof(*Awhole));
        Bwhole = malloc(N * N * sizeof(*Bwhole));
        Cwhole = malloc(N * N * sizeof(*Cwhole));
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                Awhole[i * N + j] = get_rand();
                Bwhole[i * N + j] = get_rand();
            }
        }
        // Send submatrices to other processes
        for (size_t i = 0; i < P_SQRT; ++i) {
            size_t j = i == 0 ? 1 : 0;
            for (; j < P_SQRT; ++j) {
                size_t k = (i + j) % P_SQRT;
                // send A[i][k] and B[k][j] to process (i, j)
                MPI_Send(&Awhole[i * n1 + k * n2], subsize, MPI_DOUBLE,
                        i * P_SQRT + j, 0, MPI_COMM_WORLD);
                MPI_Send(&Bwhole[k * n1 + j * n2], subsize, MPI_DOUBLE,
                        i * P_SQRT + j, 0, MPI_COMM_WORLD);
            }
        }
        // Get my own submatrix
        for (size_t i = 0; i < N_SUB; ++i) {
            for (size_t j = 0; j < N_SUB; ++j) {
                A[i * N_SUB + j] = Awhole[i * N_SUB + j];
                B[i * N_SUB + j] = Bwhole[i * N_SUB + j];
            }
        }
    } else {
        MPI_Recv(A, subsize, MPI_DOUBLE,
                0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(B, subsize, MPI_DOUBLE,
                0, 0, MPI_COMM_WORLD, &status);
    }

    /* Debugging code
    printf("My process id: %d\n", procid);
    printf("My matrix A:\n");
    print_matrix(A, N_SUB);
    printf("My matrix B:\n");
    print_matrix(B, N_SUB);
    putchar('\n');
    fflush(stdout);
    */

    /*                */
    /* Main algorithm */
    /*                */

    double start_time;
    if (procid == 0) {
        start_time = MPI_Wtime();
    }
    
    // ids of neighbors
    int up, down, left, right;
    {
        // Get my (i, j)
        size_t i = procid / P_SQRT;
        size_t j = procid % P_SQRT;
        // Calculate ids of neighbors
        up = (i == 0 ? P_SQRT - 1 : i - 1) * P_SQRT + j;
        down = (i == P_SQRT - 1 ? 0 : i + 1) * P_SQRT + j;
        left = i * P_SQRT + (j == 0 ? P_SQRT - 1 : j - 1);
        right = i * P_SQRT + (j == P_SQRT - 1 ? 0 : j + 1);
    }
    // Initialize C
    for (size_t i = 0; i < N_SUB; ++i) {
        for (size_t j = 0; j < N_SUB; ++j) {
            C[i * N_SUB + j] = 0.0;
        }
    }
    for (size_t l = 0; l < P_SQRT - 1; ++l) {
        multiply_matrices(A, B, C, N_SUB);
        MPI_Send(A, subsize, MPI_DOUBLE, left, 0, MPI_COMM_WORLD);
        MPI_Send(B, subsize, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
        MPI_Recv(Atemp, subsize, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(Btemp, subsize, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &status);
        // swap matrices with their temporary counterparts
        double *temp_ptr = A;
        A = Atemp;
        Atemp = temp_ptr;
        temp_ptr = B;
        B = Btemp;
        Btemp = temp_ptr;
    }
    multiply_matrices(A, B, C, N_SUB);

    /* Debugging code */
    if (procid == 0) {
        printf("Intermediate time elapsed: %f\n", MPI_Wtime() - start_time);
        fflush(stdout);
    }

    // Receive results from all processes
    if (procid == 0) {
        for (size_t i = 0; i < P_SQRT; ++i) {
            size_t j = i == 0 ? 1 : 0;
            for (; j < P_SQRT; ++j) {
                MPI_Recv(&Cwhole[i * n1 + j * n2], subsize, MPI_DOUBLE,
                        i * P_SQRT + j, 0, MPI_COMM_WORLD, &status);
            }
        }
        // receive my own submatrix
        for (size_t i = 0; i < N_SUB; ++i) {
            for (size_t j = 0; j < N_SUB; ++j) {
                Cwhole[i * N_SUB + j] = C[i * N_SUB + j];
            }
        }
    } else {
        MPI_Send(C, subsize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (procid == 0) {
        double el_time = MPI_Wtime() - start_time;
        printf("Time elapsed: %f\n", el_time);
        fflush(stdout);
    }

#ifdef TEST
    if (procid == 0) {
        double *Aplain = malloc(N * N * sizeof(*Aplain));
        blocks_to_plain(Awhole, Aplain, N, N_SUB, P_SQRT);
        double *Bplain = malloc(N * N * sizeof(*Bplain));
        blocks_to_plain(Bwhole, Bplain, N, N_SUB, P_SQRT);
        double *Cplain = malloc(N * N * sizeof(*Bplain));
        blocks_to_plain(Cwhole, Cplain, N, N_SUB, P_SQRT);
        double *Ctest = malloc(N * N * sizeof(*Ctest));

        /* Debugging code
        printf("Real matrix A:\n");
        print_matrix(Aplain, N);
        printf("Real matrix B:\n");
        print_matrix(Bplain, N);
        fflush(stdout);
        */

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double sum = 0;
                for (size_t k = 0; k < N; ++k) {
                    sum += Aplain[i * N + k] * Bplain[k * N + j];
                }
                Ctest[i * N + j] = sum;
            }
        }

        // compare matrices Ctest and Cplain
        int equal = 1;
        const double eps = 1.0e-7;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double diff = Ctest[i * N + j] - Cplain[i * N + j];
                if (diff < 0) {
                    diff = -diff;
                }
                if (diff >= eps) {
                    equal = 0;
                    goto loop_break;
                }
            }
        }
loop_break:
        /* Debugging code
        printf("Matrix A:\n");
        print_matrix(Aplain, N);
        printf("Matrix B:\n");
        print_matrix(Bplain, N);
        printf("Matrix C:\n");
        print_matrix(Cplain, N);
        printf("Matrix Ctest:\n");
        print_matrix(Ctest, N);
        */

        puts(equal ? "Matrices equal" : "Matrices are NOT equal");
        fflush(stdout);

        free(Aplain);
        free(Bplain);
        free(Cplain);
        free(Ctest);
    }
#endif

    if (procid == 0) {
        free(Awhole);
        free(Bwhole);
        free(Cwhole);
    }
    free(A);
    free(Atemp);
    free(B);
    free(Btemp);
    free(C);
    
    MPI_Finalize();

    return 0;
}
