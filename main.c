#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Size of a matrix
#ifndef N
#define N 1000
#endif

#ifndef P_SQRT
#define P_SQRT 2
#endif

// Number of processes
#define P (P_SQRT * P_SQRT)
// Size of a submatrix
#define N_SUB (N / P_SQRT)

// Computing C = A * B
double A[P_SQRT][P_SQRT][N_SUB][N_SUB];
double B[P_SQRT][P_SQRT][N_SUB][N_SUB];
double C[P_SQRT][P_SQRT][N_SUB][N_SUB];

double myC[N_SUB][N_SUB];

size_t indA[P_SQRT][P_SQRT][2];
size_t indB[P_SQRT][P_SQRT][2];

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
blocks_to_plain(double A[P_SQRT][P_SQRT][N_SUB][N_SUB], double *Aplain)
{
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Aplain[i * N + j] = A[i / N_SUB][j / N_SUB][i % N_SUB][j % N_SUB];
        }
    }
}

void
multiply_matrices(double A[N_SUB][N_SUB], double B[N_SUB][N_SUB], double C[N_SUB][N_SUB])
{
    for (size_t i = 0; i < N_SUB; ++i) {
        for (size_t j = 0; j < N_SUB; ++j) {
            for (size_t k = 0; k < N_SUB; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

#ifdef TEST
void
print_matrix(double *A)
{
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            printf("%f ", A[i * N + j]);
        }
        putchar('\n');
    }
}
#endif

int main()
{
    /*
    printf("P_SQRT = %d\n", P_SQRT);
    printf("P = %d\n", P);
    printf("N = %d\n", N);
    printf("N_SUB = %d\n", N_SUB);
    */

    /* Initialization */
    // Can't parallel this (rand() is thread-safe => threads block each other)
    for (size_t m = 0;  m < P_SQRT; ++m) {
        for (size_t n = 0; n < P_SQRT; ++n) {
            for (size_t i = 0; i < N_SUB; ++i) {
                for (size_t j = 0; j < N_SUB; ++j) {
                    A[m][n][i][j] = get_rand();
                    B[m][n][i][j] = get_rand();
                    C[m][n][i][j] = 0.0;
                }   
            }
        }
    }

    // Shift A and B virtually
    // rows of A by i to the left
    // rows of B by j to the right
    for (size_t i = 0; i < P_SQRT; ++i) {
        for (size_t j = 0; j < P_SQRT; ++j) {
            size_t k = (i + j) % P_SQRT;
            indA[i][j][0] = i;
            indA[i][j][1] = k;
            indB[i][j][0] = k;
            indB[i][j][1] = j;
        }
    }

    // Init myC with zeroes
    for (size_t i = 0; i < N_SUB; ++i) {
        for (size_t j = 0; j < N_SUB; ++j) {
            myC[i][j] = 0.0;
        }
    }

    double start_time = omp_get_wtime();
    #pragma omp parallel for collapse(2) firstprivate(indA, indB, myC) num_threads(P)
    for (size_t i = 0; i < P_SQRT; ++i) {
        for (size_t j = 0; j < P_SQRT; ++j) {
            // thread block
            for (size_t k = 0; k < P_SQRT; ++k) {
                size_t Ai = indA[i][j][0];
                size_t Aj = indA[i][j][1];
                size_t Bi = indB[i][j][0];
                size_t Bj = indB[i][j][1];
                multiply_matrices(A[Ai][Aj], B[Bi][Bj], myC);
                // virtual shift by 1 (A left, B up)
                for (size_t i1 = 0; i1 < P_SQRT; ++i1) {
                    size_t temp = indA[i1][0][1];
                    for (size_t j1 = 0; j1 < P_SQRT - 1; ++j1) {
                        indA[i1][j1][1] = indA[i1][j1 + 1][1];
                    }
                    indA[i1][P_SQRT - 1][1] = temp;
                }
                for (size_t j1 = 0; j1 < P_SQRT; ++j1) {
                    size_t temp = indB[0][j1][0];
                    for (size_t i1 = 0; i1 < P_SQRT - 1; ++i1) {
                        indB[i1][j1][0] = indB[i1 + 1][j1][0];
                    }
                    indB[P_SQRT - 1][j1][0] = temp;
                }
            }

            // Write myC to C[i][j]
            for (size_t i1 = 0; i1 < N_SUB; ++i1) {
                for (size_t j1 = 0; j1 < N_SUB; ++j1) {
                    C[i][j][i1][j1] = myC[i1][j1];
                }
            }
        }
    }
    double el_time = omp_get_wtime() - start_time;
    printf("Time elapsed: %f\n", el_time);

#ifdef TEST
    double *Aplain = malloc(N * N * sizeof(*Aplain));
    blocks_to_plain(A, Aplain);
    double *Bplain = malloc(N * N * sizeof(*Bplain));
    blocks_to_plain(B, Bplain);
    double *Cplain = malloc(N * N * sizeof(*Cplain));
    blocks_to_plain(C, Cplain);
    double *Ctest = malloc(N * N * sizeof(*Ctest));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0;
            for (size_t k = 0; k < N; ++k) {
                sum += Aplain[i * N + k] * Bplain[k * N + j];
            }
            Ctest[i * N +j] = sum;
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
    /*
    printf("Matrix A:\n");
    print_matrix(Aplain);
    printf("Matrix B:\n");
    print_matrix(Bplain);
    printf("Matrix C:\n");
    print_matrix(Cplain);
    printf("Matrix Ctest:\n");
    print_matrix(Ctest);
    */

    puts(equal ? "Matrices equal" : "Matrices are NOT equal");
#endif
}
