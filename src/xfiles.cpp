#include "xfiles.h"


void write_ofile_csv(const MPI_File fh, const uint iteration, const VectorXd sigmaG, const double sigmaE, const VectorXi m0,
                     const uint n_thinned_saved, const MatrixXd estPi) {
    
    MPI_Status status;
    
    char buff[LENBUF];

    int cx = snprintf(buff, LENBUF, "%5d, %4d", iteration, (int) sigmaG.size());
    assert(cx >= 0 && cx < LENBUF);
        
    for(int jj = 0; jj < sigmaG.size(); ++jj){
        cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", sigmaG(jj));
        assert(cx >= 0 && cx < LENBUF - strlen(buff));
    }
        
    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f, %20.15f, %7d, %4d, %2d",  sigmaE, sigmaG.sum()/(sigmaE+sigmaG.sum()), m0.sum(), int(estPi.rows()), int(estPi.cols()));
    assert(cx >= 0 && cx < LENBUF - strlen(buff));
        
    for (int ii=0; ii<estPi.rows(); ++ii) {
        for(int kk = 0; kk < estPi.cols(); ++kk) {
            cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", estPi(ii,kk));
            assert(cx >= 0 && cx < LENBUF - strlen(buff));
        }
    }
        
    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), "\n");
    assert(cx >= 0 && cx < LENBUF - strlen(buff));
        
    MPI_Offset offset = size_t(n_thinned_saved) * strlen(buff);
    check_mpi(MPI_File_write_at(fh, offset, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);
}
