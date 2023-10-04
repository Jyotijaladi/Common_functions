#include <stdio.h>
#include <stdlib.h>

//#14 Function to calculate prime factors of a number
int* calculatePrimeFactors(int num) {
    int i;
    int *arr=(int*)malloc(num * sizeof(int));;
    int index=0;
    
    for (i = 2; i <= num; i++) {
        while (num % i == 0) {
            arr[index++]=i;
                        num /= i;
        }
    }
    arr[index]=-1;
    return arr;
}
