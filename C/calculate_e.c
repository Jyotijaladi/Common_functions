#include <stdio.h>
#include <math.h>

int factorial(int num) {
    if (num <= 1)
        return 1;
    return num * factorial(num - 1);
}
//#7 Function to calculate the value of e
double calculateE() {
    double e = 1.0;
    int i;
    for (i = 1; i <= 10; i++) {
        e += 1.0 / factorial(i);
    }
    return e;
}
