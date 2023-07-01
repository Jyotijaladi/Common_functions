#include <stdio.h>
#include <math.h>

//#14 Function to calculate prime factors of a number
void calculatePrimeFactors(int num) {
    int i;
    printf("Prime factors of %d are: ", num);
    for (i = 2; i <= num; i++) {
        while (num % i == 0) {
            printf("%d ", i);
            num /= i;
        }
    }
}
