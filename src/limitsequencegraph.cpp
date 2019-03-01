#include "limitsequencegraph.hpp"

#include "compression.h"
#include "BayesRRmz.hpp"
#include <iostream>

LimitSequenceGraph::LimitSequenceGraph(BayesRRmz *bayes, size_t maxParallel)
    : AnalysisGraph(bayes, maxParallel)
    , m_graph(new graph)
{
    // Decompress the column for this marker
    auto f = [this] (Message msg) -> Message {
        //std::cout << "Decompress column " << msg.id << " " << msg.marker << std::endl;

        const unsigned int colSize = msg.numInds * sizeof(double);
        msg.decompressBuffer = new unsigned char[colSize];

        extractData(reinterpret_cast<unsigned char *>(m_bayes->m_data.ppBedMap) + m_bayes->m_data.ppbedIndex[msg.marker].pos,
                static_cast<unsigned int>(m_bayes->m_data.ppbedIndex[msg.marker].size),
                msg.decompressBuffer,
                colSize);

        return msg;
    };
    // Do the decompression work on up to maxParallel threads at once
    m_decompressNode.reset(new function_node<Message, Message>(*m_graph, m_maxParallel, f));

    // The sequencer node enforces the correct ordering based upon the message id
    m_ordering.reset(new sequencer_node<Message>(*m_graph, [] (const Message& msg) -> unsigned int {
        return msg.id;
    }));

    m_ordering2.reset(new sequencer_node<Message>(*m_graph, [] (const Message& msg) -> unsigned int {
        return msg.id;
    }));

    // Do not allow predecessors to carry on blindly until later parts of
    // the graph have finished and freed up some resources.
    m_limit.reset(new limiter_node<Message>(*m_graph, m_maxParallel));

    auto g = [this] (Message msg) -> continue_msg {
        //std::cout << "Sampling for id: " << msg.id << std::endl;

        // Delegate the processing of this column to the algorithm class
        Map<VectorXf> Cx(reinterpret_cast<float *>(msg.decompressBuffer), msg.numInds);
        m_bayes->processColumn(msg.marker, Cx);

        // Cleanup
        delete[] msg.decompressBuffer;
        msg.decompressBuffer = nullptr;

        // Signal for next decompression task to continue
        return continue_msg();
    };
    // The sampling node is enforced to behave in a serial manner to ensure that the resulting chain
    // is ergodic.
    m_samplingNode.reset(new function_node<Message>(*m_graph, serial, g));

    // Set up the graph topology:
    //
    // orderingNode -> limitNode -> decompressionNode (parallel) -> orderingNode -> samplingNode (sequential)
    //                      ^                                                           |
    //                      |___________________________________________________________|
    //
    // This ensures that we always run the samplingNode on the correct order of markers and signal back to
    // the parallel decompression to keep it constantly fed. This should be a self-balancing graph.
    make_edge(*m_ordering, *m_limit);
    make_edge(*m_limit, *m_decompressNode);
    make_edge(*m_decompressNode, *m_ordering2);
    make_edge(*m_ordering2, *m_samplingNode);

    // Feedback that we can now decompress another column
    make_edge(*m_samplingNode, m_limit->decrement);
}

void LimitSequenceGraph::exec(unsigned int numInds,
                              unsigned int numSnps,
                              const std::vector<unsigned int> &markerIndices)
{
    // Reset the graph from the previous iteration. This resets the sequencer node current index etc.
    m_graph->reset();

    // Push some messages into the top of the graph to be processed - representing the column indices
    for (unsigned int i = 0; i < numSnps; ++i) {
        Message msg;
        msg.id = i;
        msg.marker = markerIndices[i];
        msg.numInds = numInds;
        m_ordering->try_put(msg);
    }

    // Wait for the graph to complete
    m_graph->wait_for_all();
}
