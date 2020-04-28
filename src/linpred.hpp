#include "data.hpp"
#include "options.hpp"

class LinPred {
    public:
        Data &data;
        Options &opt;

        LinPred(Data &data, Options &opt);

        void predict_genetic_values(string outfile);

private:
    Eigen::IOFormat csvFormat;
};