#include <stdio.h>
#include <math.h>

//#4 Function to calculate the factorial of a number
int factorial(int num) {
    if (num <= 1)
        return 1;
    return num * factorial(num - 1);
}

//#5 Function to calculate cos(x) value
double calculateCos(double x) {
    double cosValue = 0.0;
    int i;
    for (i = 0; i <= 10; i++) {
        int sign = (i % 2 == 0) ? 1 : -1;
        double term = sign * pow(x, 2 * i) / factorial(2 * i);
        cosValue += term;
    }
    return cosValue;
}

