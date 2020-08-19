#
# Makefile for compiling the auxiliary scripts for postprocessing
# Daniel Trejo Banos
#

objects: beta_checker beta_converter components_converter epsilon_converter extract_non_zero_betaAll  extract_non_zero_cpnAll_test
all: $(objects)
$(objects): %: %.c
	$(CXX) $(CXXFLAGS) -o $@ $<
