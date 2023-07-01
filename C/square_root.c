#include <stdio.h>
#include <math.h>

//#9 Function to calculate square root of a number
double squareRoot(double num) {
    double guess = 1.0;
    while (fabs(guess * guess - num) >= 0.0001) {
        guess = (num / guess + guess) / 2.0;
    }
    return guess;
}