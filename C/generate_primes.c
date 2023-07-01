
#include <stdio.h>
#include <math.h>

//#8 Function to check if a number is prime
int isPrime(int num) {
    int i;
    if (num <= 1)
        return 0;
    for (i = 2; i <= sqrt(num); i++) {
        if (num % i == 0)
            return 0;
    }
    return 1;
}

//#8 Function to generate prime numbers
void generatePrimes(int n) {
    int count = 0, num = 2;
    while (count < n) {
        if (isPrime(num)) {
            printf("%d ", num);
            count++;
        }
        num++;
    }
}
