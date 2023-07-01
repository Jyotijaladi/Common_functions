#include <stdio.h>
#include <math.h>

//#1 Function to generate the nth Fibonacci number
int fibonacci(int n) {
    if (n <= 1)
        return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

//#2 Function to reverse a decimal number
int reverseDecimal(int num) {
    int reversed = 0;
    while (num > 0) {
        reversed = reversed * 10 + num % 10;
        num /= 10;
    }
    return reversed;
}

//#3 Function to convert binary number to decimal number
int binaryToDecimal(int binary) {
    int decimal = 0, base = 1;
    while (binary > 0) {
        decimal += (binary % 10) * base;
        binary /= 10;
        base *= 2;
    }
    return decimal;
}
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



//#6 Function to calculate the GCD of two numbers
int calculateGCD(int a, int b) {
    if (b == 0)
        return a;
    return calculateGCD(b, a % b);
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

//#9 Function to calculate square root of a number
double squareRoot(double num) {
    double guess = 1.0;
    while (fabs(guess * guess - num) >= 0.0001) {
        guess = (num / guess + guess) / 2.0;
    }
    return guess;
}

//#10 Function to convert character to ASCII code
int charToAscii(char c) {
    return (int)c;
}


//#11 Function to swap two elements
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

//#11 Function to partition an array
int partitionArray(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    int j;
    for (j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}


//#12 Function to remove duplicate elements from an array
int removeDuplicates(int arr[], int n) {
    int i, j, k;
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n;) {
            if (arr[j] == arr[i]) {
                for (k = j; k < n - 1; k++)
                    arr[k] = arr[k + 1];
                n--;
            }
            else
                j++;
        }
    }
    return n;
}

//#13 Function to count duplicate elements in an array
int countDuplicates(int arr[], int n) {
    int count = 0, i, j;
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            if (arr[j] == arr[i]) {
                count++;
                break;
            }
        }
    }
    return count;
}

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

//#15 Function to reverse an array
void reverseArray(int arr[], int n) {
    int i, temp;
    for (i = 0; i < n / 2; i++) {
        temp = arr[i];
        arr[i] = arr[n - i - 1];
        arr[n - i - 1] = temp;
    }
}

//#16 Function to find the Kth smallest number in an array
int findKthSmallest(int arr[], int n, int k) {
    int i, j, temp;
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            if (arr[j] < arr[i]) {
                temp = arr[j];
                arr[j] = arr[i];
                arr[i] = temp;
            }
        }
    }
    return arr[k - 1];
}

//#17 Function to implement merge sort
void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

//#18 Function to implement bubble sort
void bubbleSort(int arr[], int n) {
    int i, j;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

//#19 Function to implement selection sort
void selectionSort(int arr[], int n) {
    int i, j, min_idx;
    for (i = 0; i < n - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        swap(&arr[min_idx], &arr[i]);
    }
}

//#20 Function to implement binary search
int binarySearch(int arr[], int low, int high, int target) {
    if (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target)
            return mid;
        if (arr[mid] > target)
            return binarySearch(arr, low, mid - 1, target);
        return binarySearch(arr, mid + 1, high, target);
    }
    return -1;
}
