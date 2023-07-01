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
