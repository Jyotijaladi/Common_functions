

//#6 Function to calculate the GCD of two numbers
int calculateGCD(int a, int b) {
    if (b == 0)
        return a;
    return calculateGCD(b, a % b);
}