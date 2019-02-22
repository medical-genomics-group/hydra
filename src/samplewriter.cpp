#include "samplewriter.h"

#include <iostream>

using namespace Eigen;

SampleWriter::SampleWriter()
    : m_commaInitFormat(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "")
{
}

SampleWriter::~SampleWriter()
{
    close();
}

void SampleWriter::open()
{
    std::cout << "Opening results file " << m_fileName << std::endl;
    m_outFile.open(m_fileName);

    m_outFile << "iteration," << "mu,";
    for (unsigned int i = 0; i < m_markerCount; ++i) {
        m_outFile << "beta[" << (i+1) << "],";
    }

    m_outFile << "sigmaE," << "sigmaG,";
    for (unsigned int i = 0; i < m_markerCount; ++i) {
        m_outFile << "comp[" << (i+1) << "],";
    }

    unsigned int i;
    for (i = 0; i < (m_individualCount - 1); ++i) {
        m_outFile << "epsilon[" << (i + 1) << "],";
    }

    m_outFile << "epsilon[" << (i + 1) << "]";
    m_outFile << std::endl;
    m_outFile.flush();
}

void SampleWriter::write(const Eigen::VectorXd &sample)
{
    //std::cout << "Writing sample" << std::endl;
    m_outFile << sample.transpose().format(m_commaInitFormat) << std::endl;
    m_outFile.flush();
}

void SampleWriter::close()
{
    m_outFile.close();
}
