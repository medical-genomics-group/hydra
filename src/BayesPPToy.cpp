#include "BayesPPToy.hpp"
#include <numeric> // For std::iota

BayesPPToy::BayesPPToy(Data &data)
    : data(data)
{
}

BayesPPToy::~BayesPPToy()
{
}

void BayesPPToy::runToyExample(int samples)
{
    std::cout << "Running preprocessed toy example ";

    const size_t markerCount = static_cast<size_t>(data.mappedZ.cols());
    std::cout << "Sampling " << markerCount << " snps\n";

    // Generate vector of indices using std::iota (don't reinvent the wheel and make intent clearer)
    std::vector<int> markerIndices(markerCount);
    std::iota(markerIndices.begin(), markerIndices.end(), 0);

    // Track progress
    const auto workItems = samples * markerCount;
    size_t currentWorkItem = 0;

    for (int i = 0; i < samples; ++i) {
        std::random_shuffle(markerIndices.begin(), markerIndices.end());

        for (size_t j = 0; j < markerCount; ++j) {
            ++currentWorkItem;

            // Calculate stats *every* work item
            const int marker = markerIndices[j];
            const double mean = double(data.mappedZ.col(marker).mean());
            const double min = double(data.mappedZ.col(marker).minCoeff());
            const double max = double(data.mappedZ.col(marker).maxCoeff());
            const double sqNorm = double(data.mappedZPZDiag[marker]);
            const auto size = data.mappedZ.col(marker).size();

            // Display progress at regular intervals
            if (j % 100 == 0) {
                printf("%3.2f%%  -> marker %6i has mean %13.6f on %ld elements [%13.6f, %13.6f]  Sq. Norm = %13.6f\n",
                       100.0 * double(currentWorkItem) / double(workItems),
                       marker,
                       mean,
                       size,
                       min,
                       max,
                       sqNorm);
                fflush(stdout);
            }
        }
    }

    std::cout << "BayesPPToy success" << endl;
}
