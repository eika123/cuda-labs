#include <stdio.h>


#define MAX(x, y) ((x) > (y)) ? (x) : (y)

int main(int argc, char** argv) {
        if (argc < 2) {
                fprintf(stderr,
                        "Usage: %s <N>\nN is the number of threads to run",
                        argv[0]);
                exit(1);
        }
        int N = atoi(argv[1]);
        printf("N = %d\n", N);


}
