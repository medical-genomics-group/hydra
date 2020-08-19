// convert components file to text file, created by AK
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


int main(int argc, const char * argv[]) {

    if (argc != 5) {
        printf("Wrong number of arguments passed: %d; expected 4 (path to .cpn file, path to .bet file, min iteration, max iteration to convert)!\n", argc - 1);
        exit(1);
    }

    FILE* cpnfile = fopen(argv[1], "rb");
    if (cpnfile == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        exit(1);
    }
    FILE* betfile = fopen(argv[2], "rb");
    if (betfile == NULL) {
        printf("Error opening file: %s\n", argv[2]);
        exit(1);
    }

    uint min_iter   = atoi(argv[3]);
    uint max_iter   = atoi(argv[4]);

    uint M = 0;
    fread(&M, sizeof(uint), 1, betfile);
    fclose(betfile);

    int cpn;
    for(uint iter = min_iter; iter <= max_iter; iter++){

        for (uint marker=0; marker < M; ++marker) {
	    
	unsigned long long int location = (unsigned long long int)sizeof(uint) + (unsigned long long int)sizeof(uint) * (unsigned long long int)(iter+1) + (unsigned long long int)sizeof(int) * ((unsigned long long int)M *  (unsigned long long int)iter + (unsigned long long int)marker);

            fseek(cpnfile, location, SEEK_SET);
            fread(&cpn, sizeof(int), 1, cpnfile);
	    //@@ DT: I suggest saving the non-zero locations found here and then read the non-zero betas

	    if(cpn > 0){
            	printf("%7d %7d %2d\n",iter , marker, cpn);
            }
	}
    }
    fclose(cpnfile);

    return 0;
}
