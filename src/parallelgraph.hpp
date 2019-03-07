#ifndef PARALLELGRAPH_H
#define PARALLELGRAPH_H

#include "analysisgraph.hpp"

#include "tbb/flow_graph.h"
#include <functional>
#include <memory>

class BayesRRmz;

using namespace tbb::flow;

class ParallelGraph : public AnalysisGraph
{
public:
    ParallelGraph(BayesRRmz *bayes, size_t maxParallel = 6);


    void exec(unsigned int numInds,
              unsigned int numSnps,
              const std::vector<unsigned int> &markerIndices) override;

private:
    struct Message {
        unsigned int id;
        unsigned int marker;
        unsigned int numInds;
    };

    std::unique_ptr<graph> m_graph;
    std::unique_ptr<function_node<Message>> m_computeNode;
    std::unique_ptr<limiter_node<Message>> m_limit;
    std::unique_ptr<sequencer_node<Message>> m_ordering;
};

#endif // PARALLELGRAPH_H
