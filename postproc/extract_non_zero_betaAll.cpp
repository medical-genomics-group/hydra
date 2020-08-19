// extract non-zero betas, created by AK
//
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


int main(int argc, const char * argv[]) {

    if (argc != 4) {
        printf("Wrong number of arguments passed: %d; expected 3 (path to .bet file, min iteration, max iteration to convert)!\n", argc - 1);
        exit(1);
    }

    FILE* betafile = fopen(argv[1], "rb");
    if (betafile == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        exit(1);
    }

    uint min_iter   = atoi(argv[2]);
    uint max_iter   = atoi(argv[3]);

    uint M = 0;
    fread(&M, sizeof(uint), 1, betafile);

    double beta;
    uint   itthin;
    size_t offset;

    for(uint iter = min_iter; iter <= max_iter; iter++){
        offset = sizeof(uint) + iter * (sizeof(uint) + M * sizeof(double));
        fseek(betafile, offset, SEEK_SET);
        fread(&itthin, sizeof(uint), 1, betafile);

        for (uint marker=0; marker < M; ++marker) {
	    
	unsigned long long int location = (unsigned long long int)sizeof(uint) + (unsigned long long int)sizeof(uint) * (unsigned long long int)(iter+1) + (unsigned long long int)sizeof(double) * ((unsigned long long int)M *  (unsigned long long int)iter + (unsigned long long int)marker);

            fseek(betafile, location, SEEK_SET);
            fread(&beta, sizeof(double), 1, betafile);
	    //@@ DT: We have to change this, to use cpn file first to select non-zero betas
	    if(beta > 0.00000000000000001 || beta < -0.00000000000000001){
            	printf("%7d %7d %20.12f\n",iter , marker, beta);
            }
	}
    }
    fclose(betafile);

    return 0;
}
