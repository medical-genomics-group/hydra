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
        Message(unsigned int id = 0, unsigned int marker = 0, unsigned int numInds = 0)
            : id(id)
            , marker(marker)
            , numInds(numInds)
        {

        }

        unsigned int id = 0;
        unsigned int marker = 0;
        unsigned int numInds = 0;

        using DataPtr = std::shared_ptr<unsigned char[]>;
        DataPtr data = nullptr;

        double beta = 0.0;
    };

    std::unique_ptr<graph> m_graph;
    std::unique_ptr<function_node<Message, Message>> m_asyncComputeNode;
    std::unique_ptr<limiter_node<Message>> m_limit;
    std::unique_ptr<sequencer_node<Message>> m_ordering;

    using decision_node = multifunction_node<Message, tbb::flow::tuple<continue_msg, Message> >;
    std::unique_ptr<decision_node> m_decisionNode;
    std::unique_ptr<function_node<Message>> m_globalComputeNode;
};

#endif // PARALLELGRAPH_H
