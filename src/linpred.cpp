#include "linpred.hpp"
#include "data.hpp"
#include "options.hpp"
#include <mpi.h>
#include "mpi_utils.hpp"

LinPred::LinPred(Data &data, Options &opt)
    : data(data)
    , opt(opt)
    , csvFormat(StreamPrecision, DontAlignCols, ", ", "\n")
{
}

/* Computes prediction for each individual from pre-computed effect estimates
 * pre  : binary PLINK files and .bet files have been read and processed
 * post : NxI prediction matrix is stored in pred member variable
 */
void LinPred::predict_genetic_values(string outfile) {
    // perform prediction
    data.pred.resize(data.Z_common.rows(), data.predBet.cols());
    // scatter rows and cols across processes
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // get data dimensions and create arrays
    int N = data.Z_common.rows();
    int M = data.Z_common.cols();
    int I = data.predBet.cols();
    printf("N = %d, M = %d, I = %d\n", N, M, I);
    float a[N*M];
    float b[M*I];
    // map data matrices to arrays for scattering
    Map<MatrixXf>(a, N, M);
    Map<MatrixXf>(b, M, I);
    // assume N and I are divisible by block size
    const uint n_processors = 4;
    int block_rows_a = N / n_processors;
    int block_cols_b = I / n_processors;
    printf("blocks of rows in A: %d\nblocks of cols in B: %d\n", block_rows_a, block_cols_b);
    // get number of elements per block
    uint elem_per_proc_a = block_rows_a * M;
    uint elem_per_proc_b = block_cols_b * M;
    float *buff_a = (float *) malloc(sizeof(float) * elem_per_proc_a);
    float *buff_b = (float *) malloc(sizeof(float) * elem_per_proc_b);
    float *buff_c = (float *) malloc(sizeof(float) * block_rows_a * block_cols_b);
    // scatter unique blocks to processors
    printf("INFO: Scattering %d elements of A across %d tasks\n", elem_per_proc_a, nranks);
    MPI_Scatter(a, elem_per_proc_a, MPI_FLOAT, buff_a,
                elem_per_proc_a, MPI_FLOAT, 0, MPI_COMM_WORLD);
    printf("INFO: Scattering %d elements of B across %d tasks\n", elem_per_proc_b, nranks);
    MPI_Scatter(b, elem_per_proc_b, MPI_FLOAT, buff_b,
                elem_per_proc_b, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // perform multiplication
    for (uint i = 0; i < block_rows_a; i++) {
        for (uint j = 0; j < block_cols_b; j++) {
            for (uint k = 0; k < M; k++) {
                buff_c[i + j * block_rows_a] += buff_a[i*M + k] * buff_b[k*M + j];
            }
        }
    }
    // gather results
    float *c = NULL;
    if (rank == 0) {
        // TODO: initialise result vector
        c = (float *) malloc(sizeof(float) * N * I);
    }
    MPI_Gather(buff_c, block_rows_a * block_cols_b, MPI_FLOAT, c,
                block_rows_a * block_cols_b, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (uint i = 0; i < N; i++) {
        for (uint j = 0; j < I; j++) {
            data.pred(i, j) = c[i*N + j];
        }
    }
    // TODO: Map C back to a matrix
    // write prediction matrix to disk
    // TODO: refactor to a writer function
    ofstream file(outfile.c_str());
    file << data.pred.format(csvFormat) << std::endl;
    file.flush();
    cout << "Predictions written to disk" << std::endl;
}
