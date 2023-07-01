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
