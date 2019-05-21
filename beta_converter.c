#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char * argv[]) {
    printf("__CONVERT BETA FILES TO TEXT__\n");

    if (argc != 3) {
        printf("Wrong number of arguments passed: %d; expected 3 (path to .bet file, and number of iterations to convert)!\n", argc - 1);
        exit(1);
    }

    FILE* fh = fopen(argv[1], "rb");
    if (fh == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        exit(1);
    }

    uint niter   = atoi(argv[2]);

    uint M = 0;
    fread(&M, sizeof(uint), 1, fh);
    printf("%d markers were processed.\n", M);

    double beta;
    for (uint iter=0; iter<niter; ++iter) {
        for (uint marker=0; marker<M; ++marker) {
            fseek(fh, sizeof(uint) + sizeof(double) * (M * iter + marker), SEEK_SET);
            fread(&beta, sizeof(double), 1, fh);
            printf("%5d/%7d = %15.10f\n", iter, marker, beta);
        }
    }

    fclose(fh);

    return 0;
}
