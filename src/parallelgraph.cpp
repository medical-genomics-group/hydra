#include "parallelgraph.hpp"

#include "compression.h"
#include "BayesRRmz.hpp"

#include <iostream>

ParallelGraph::ParallelGraph(BayesRRmz *bayes, size_t maxParallel)
    : AnalysisGraph(bayes, maxParallel)
    , m_graph(new graph)
{
    // Decompress the column for this marker then process the column using the algorithm class
    auto f = [this] (Message msg) -> continue_msg {
        // Decompress the column
        const unsigned int colSize = msg.numInds * sizeof(double);
        unsigned char *decompressBuffer = new unsigned char[colSize];

        extractData(reinterpret_cast<unsigned char *>(m_bayes->m_data.ppBedMap) + m_bayes->m_data.ppbedIndex[msg.marker].pos,
                    static_cast<unsigned int>(m_bayes->m_data.ppbedIndex[msg.marker].size),
                    decompressBuffer,
                    colSize);

        // Delegate the processing of this column to the algorithm class
        Map<VectorXd> Cx(reinterpret_cast<double *>(decompressBuffer), msg.numInds);
        m_bayes->processColumnAsync(msg.marker, Cx);

        // Cleanup
        delete[] decompressBuffer;
        decompressBuffer = nullptr;

        // Signal for next decompression task to continue
        return continue_msg();
    };

    // Do the decompress and compute work on up to maxParallel threads at once
    m_computeNode.reset(new function_node<Message>(*m_graph, m_maxParallel, f));

    // Limit the number of parallel computations
    m_limit.reset(new limiter_node<Message>(*m_graph, m_maxParallel));

    // Enforce the correct order, based on the message id
    m_ordering.reset(new sequencer_node<Message>(*m_graph, [] (const Message& msg) -> unsigned int {
        return msg.id;
    }));

    // Set up the graph topology:
    //
    // orderingNode -> limitNode -> decompressionAndSamplingNode (parallel)
    //                      ^                   |
    //                      |___________________|
    //
    // Run the decompressionAndSampling node in the correct order, but do not wait for the most
    // up-to-date data.
    make_edge(*m_ordering, *m_limit);
    make_edge(*m_limit, *m_computeNode);

    // Feedback that we can now decompress another column
    make_edge(*m_computeNode, m_limit->decrement);
}

void ParallelGraph::exec(unsigned int numInds,
                         unsigned int numSnps,
                         const std::vector<unsigned int> &markerIndices)
{
    // Do not allow Eigen to parallalize during ParallelGraph execution.
    const auto eigenThreadCount = Eigen::nbThreads( );
    Eigen::setNbThreads(0);

    // Reset the graph from the previous iteration. This resets the sequencer node current index etc.
    m_graph->reset();

    // Push some messages into the top of the graph to be processed - representing the column indices
    for (unsigned int i = 0; i < numSnps; ++i) {
        Message msg = { i, markerIndices[i], numInds };
        m_ordering->try_put(msg);
    }

    // Wait for the graph to complete
    m_graph->wait_for_all();

    // Turn Eigen threading back on.
    Eigen::setNbThreads(eigenThreadCount);
}
