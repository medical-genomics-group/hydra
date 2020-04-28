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
    // TESTING:
    cout << "Matrix A:" << endl;
    for (uint i = 0; i < data.Z_common.rows(); i++) {
        for (uint j = 0; j < data.Z_common.cols(); j++) {
            data.Z_common(i, j) = (double) i;
            cout << data.Z_common(i, j) << " ";
        }
    cout << std::endl;
    }
    cout << "Matrix B:" << endl;
    for (uint i = 0; i < data.predBet.rows(); i++) {
        for (uint j = 0; j < data.predBet.cols(); j++) {
            data.predBet(i, j) = (double) i * j;
            cout << data.predBet(i, j) << " ";
        }
    cout << std::endl;
    }
    // perform prediction
    data.pred.resize(data.Z_common.rows(), data.predBet.cols());
    // scatter rows and cols across processes
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "nranks: " << nranks << std::endl;

    // get data dimensions and create arrays
    int N = data.Z_common.rows();
    int M = data.Z_common.cols();
    int I = data.predBet.cols();
    printf("N = %d, M = %d, I = %d\n", N, M, I);
    double a[N*M];
    double b[M*I];
    // map data matrices to arrays for scattering
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(a, N, M) = data.Z_common; // A must be flattened in row-major format
    Map<MatrixXd>(b, M, I) = data.predBet;
    cout << "Elements of array a: ";
    for (int i = 0; i < N*M; i++) { cout << a[i] << " "; }
    cout << endl;
    cout << "Elements of array b: ";
    for (int i = 0; i < M*I; i++) { cout << b[i] << " "; }
    cout <<endl;
    // assume N and I are divisible by block size
    int block_rows_a = N / nranks;
    int block_cols_b = I / nranks;
    printf("blocks of rows in A: %d\nblocks of cols in B: %d\n", block_rows_a, block_cols_b);
    // get number of elements per block
    uint elem_per_proc_a = block_rows_a * M;
    uint elem_per_proc_b = block_cols_b * M;
    double *buff_a = (double *) malloc(sizeof(double) * elem_per_proc_a);
    double *buff_b = (double *) malloc(sizeof(double) * elem_per_proc_b);
    double *buff_c = (double *) malloc(sizeof(double) * block_rows_a * block_cols_b);
    // scatter unique blocks to processors
    printf("INFO: Scattering %d elements of A across %d tasks\n", elem_per_proc_a, nranks);
    MPI_Scatter(a, elem_per_proc_a, MPI_DOUBLE, buff_a,
                elem_per_proc_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout << "Elements of A: ";
    for (uint i = 0; i < elem_per_proc_a; i++) { cout <<  buff_a[i] << " "; }
    cout << std::endl;
    printf("INFO: Scattering %d elements of B across %d tasks\n", elem_per_proc_b, nranks);
    MPI_Scatter(b, elem_per_proc_b, MPI_DOUBLE, buff_b,
                elem_per_proc_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout << "Elements of B: ";
    for (uint i = 0; i < elem_per_proc_b; i++) { cout <<  buff_b[i] << " "; }
    cout << std::endl;
    // perform multiplication
    for (uint i = 0; i < block_rows_a; i++) {
        for (uint j = 0; j < block_cols_b; j++) {
            for (uint k = 0; k < M; k++) {
                buff_c[i + block_rows_a * j] += buff_a[i + block_rows_a * k] * buff_b[j * M + k];
            }
        }
    }
    // gather results
    double *c = NULL;
    if (rank == 0) {
        c = (double *) malloc(sizeof(double) * N * I);
    }
    MPI_Gather(buff_c, block_rows_a * block_cols_b, MPI_DOUBLE, c,
                block_rows_a * block_cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // TODO: map C back to a matrix directly
    if (rank == 0) {
        for (uint i = 0; i < N; i++) {
            for (uint j = 0; j < I; j++) {
                data.pred(i, j) = c[i + j * N];
            }
        cout << endl;
        }
    
        // write prediction matrix to disk
        // TODO: refactor to a writer function
        ofstream file(outfile.c_str());
        file << data.pred.format(csvFormat) << std::endl;
        file.flush();
        cout << "Predictions written to disk" << std::endl;
    }
}
