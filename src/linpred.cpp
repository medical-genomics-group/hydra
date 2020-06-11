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
    cout << data.Z << std::endl;
    // perform prediction
    data.pred.resize(data.Z.rows(), data.predBet.cols());
    // scatter rows and cols across processes
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // get data dimensions and create arrays
    int N = data.Z.rows();
    int M = data.Z.cols();
    int I = data.predBet.cols();
    printf("N = %d, M = %d, I = %d\n", N, M, I);
    double a[N*M];
    double b[M*I];
    // map data matrices to arrays for scattering
    // we'll scatter A, and broadcast B because we assume B is smaller
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(a, N, M) = data.Z; // LHS row-major format
    Map<MatrixXd>(b, M, I) = data.predBet; // RHS col-major format
    // assume N is divisible by number of processes
    int block_rows_a = N / nranks;
    // get number of elements per block
    uint elem_per_proc_a = block_rows_a * M;
    double *buff_a = (double *) malloc(sizeof(double) * elem_per_proc_a);
    double *buff_c = (double *) malloc(sizeof(double) * block_rows_a * I);
    for (uint i = 0; i < block_rows_a * I; i++) {
        buff_c[i] = 0.0;
    }
    // scatter unique blocks of A to processors
    printf("INFO: Scattering %d elements of A to each of %d tasks\n", elem_per_proc_a, nranks);
    MPI_Scatter(a, elem_per_proc_a, MPI_DOUBLE, buff_a,
                elem_per_proc_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // perform multiplication
    //  TODO: parallelise loops
    for (uint i = 0; i < block_rows_a; i++) {
        for (uint j = 0; j < I; j++) {
            for (uint k = 0; k < M; k++) {
                buff_c[i + block_rows_a * j] += buff_a[i + block_rows_a * k] * b[j * M + k];
            }
        }
    }
    // gather results
    double *c = NULL;
    if (rank == 0) {
        c = (double *) malloc(sizeof(double) * N * I);
    }
    MPI_Gather(buff_c, block_rows_a * I, MPI_DOUBLE, c,
                block_rows_a * I, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // map c array back to MatrixXd
    if (rank == 0) {
        data.pred = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(c, N, I);
    }
    // write prediction matrix to disk
    // TODO: refactor to a writer function
    ofstream file(outfile.c_str());
    file << data.pred.format(csvFormat) << std::endl;
    file.flush();
    cout << "Predictions written to disk" << std::endl;
}
