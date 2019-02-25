//
//  main.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//
// Modified by Daniel Trejo Banos for memory mapped filed implementation

#include <iostream>
#include <string>
#include "BayesRRm.h"

#include "data.hpp"
#include "options.hpp"

using namespace std;


int main(int argc, const char * argv[]) {


	cout << "***********************************************\n";
	cout << "* BayesRRcmd                                  *\n";
	cout << "* Complex Trait Genetics group UNIL            *\n";
	cout << "*                                             *\n";
	cout << "* MIT License                                 *\n";
	cout << "***********************************************\n";

	Gadget::Timer timer;
	timer.setTime();
	cout << "\nAnalysis started: " << timer.getDate();

	if (argc < 2){
		cerr << " \nDid you forget to give the input parameters?\n" << endl;
		exit(1);
	}
	try {

		Options opt;
		opt.inputOptions(argc, argv);

		Data data;

		if (opt.analysisType == "Bayes" && opt.bayesType == "bayes") {

			clock_t start = clock();

			data.readFamFile(opt.bedFile + ".fam");
			data.readBimFile(opt.bedFile + ".bim");
			data.readPhenotypeFile(opt.phenotypeFile);

			cout << "Start reading " << opt.bedFile+".bed" << endl;
			clock_t start_bed = clock();
			//data.readBedFile(opt.bedFile+".bed");
			data.readBedFile_noMPI(opt.bedFile+".bed");
			clock_t end   = clock();
			printf("Finished reading the bed file in %.3f sec.\n", (float)(end - start_bed) / CLOCKS_PER_SEC);
			cout << endl;

			//TODO non memory mapped version here

			end = clock();
			printf("OVERALL read+compute time = %.3f sec.\n", (float)(end - start) / CLOCKS_PER_SEC);

		} else if (opt.analysisType == "Bayes" && (opt.bayesType == "bayesMmap"   || opt.bayesType == "horseshoe")) {

			clock_t start = clock();

			data.readFamFile(opt.bedFile + ".fam");
			data.readBimFile(opt.bedFile + ".bim");
			data.readPhenotypeFile(opt.phenotypeFile);
			data.readBedFile_noMPI(opt.bedFile+".bed");

			if (opt.bayesType == "bayesMmap") {
				BayesRRm mmapToy(data, opt, sysconf(_SC_PAGE_SIZE));
				mmapToy.runGibbs();

			} else if (opt.bayesType == "horseshoe") {

			}

			clock_t end   = clock();
			printf("OVERALL read+compute time = %.3f sec.\n", (float)(end - start) / CLOCKS_PER_SEC);

		} else if (opt.analysisType == "Preprocess") {
			data.readFamFile(opt.bedFile + ".fam");
			data.readBimFile(opt.bedFile + ".bim");
			data.readPhenotypeFile(opt.phenotypeFile);

			cout << "Start preprocessing " << opt.bedFile + ".bed" << endl;
			clock_t start_bed = clock();
			data.preprocessBedFile(opt.bedFile + ".bed",
					opt.bedFile + ".ppbed",
					opt.bedFile + ".ppbedindex",
					opt.compress);
			clock_t end = clock();
			printf("Finished preprocessing the bed file in %.3f sec.\n", double(end - start_bed) / double(CLOCKS_PER_SEC));
			cout << endl;

		} else if (opt.analysisType == "PPBayes") {
			clock_t start = clock();

			data.readFamFile(opt.bedFile + ".fam");
			data.readBimFile(opt.bedFile + ".bim");
			data.readPhenotypeFile(opt.phenotypeFile);

			cout << "Start reading preprocessed bed file: " << opt.bedFile + ".ppbed" << endl;
			clock_t start_bed = clock();
			data.mapPreprocessBedFile(opt.bedFile + ".ppbed");
			clock_t end = clock();
			printf("Finished reading preprocessed bed file in %.3f sec.\n", double(end - start_bed) / double(CLOCKS_PER_SEC));
			cout << endl;

			// Run analysis using mapped data files
			BayesRRm toy(data, opt,sysconf(_SC_PAGE_SIZE));
			toy.runGibbs();

			data.unmapPreprocessedBedFile();
			end = clock();
			printf("OVERALL read+compute time = %.3f sec.\n", double(end - start) / double(CLOCKS_PER_SEC));
		} else {
			throw(" Error: Wrong analysis type: " + opt.analysisType);
		}
	}
	catch (const string &err_msg) {
		cerr << "\n" << err_msg << endl;
	}
	catch (const char *err_msg) {
		cerr << "\n" << err_msg << endl;
	}

	timer.getTime();

	cout << "\nAnalysis finished: " << timer.getDate();
	cout << "Computational time: "  << timer.format(timer.getElapse()) << endl;


	return 0;
}
