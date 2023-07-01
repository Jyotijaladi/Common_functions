#include <stdio.h>
#include <math.h>

//#4 Function to calculate the factorial of a number
int factorial(int num) {
    if (num <= 1)
        return 1;
    return num * factorial(num - 1);
}
//#4 Function to calculate sin(x) value
double calculateSin(double x) {
    double sinValue = 0.0;
    int i;
    for (i = 0; i <= 10; i++) {
        int sign = (i % 2 == 0) ? 1 : -1;
        double term = sign * pow(x, 2 * i + 1) / factorial(2 * i + 1);
        sinValue += term;
    }
    return sinValue;
}