#ifndef BAYESPPTOY_HPP_
#define BAYESPPTOY_HPP_

#include "data.hpp"
#include <Eigen/Eigen>

class BayesPPToy
{
	const Data &data;
public:
    BayesPPToy(Data &data);
    virtual ~BayesPPToy();

	void runToyExample(int samples);
};

#endif /* BAYESPPTOY_HPP_ */
