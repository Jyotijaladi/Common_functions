#include <stdio.h>
#include <stdlib.h>

int isPrime(int num) {
    if (num <= 1) return 0;
    if (num <= 3) return 1;
    if (num % 2 == 0 || num % 3 == 0) return 0;

    for (int i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) return 0;
    }

    return 1;
}

// Function to calculate prime factors of a number
int* calculatePrimeFactors(int num, int* primeFactorsCount) {
    int* factors = NULL;
    int count = 0;
    int i = 2;

    while (num > 1) {
        if (num % i == 0 && isPrime(i)) {
            count++;
            factors = (int*)realloc(factors, count * sizeof(int));
            factors[count - 1] = i;
            num /= i;
        } else {
            i++;
        }
    }

    *primeFactorsCount = count;

    return factors;
}
