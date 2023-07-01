import java.util.Scanner;
public class PrimeSeries {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Please provide N as a command line argument");
            return;
        }

        int n = Integer.parseInt(args[0]);

        for (int i = 2; i <= n; i++) {
            if (isPrime(i)) {
                System.out.print(i + " ");
            }
        }
        System.out.println();
    }

    public static boolean isPrime(int num) {
        if (num <= 1) {
            return false;
        }

        for (int i = 2; i <= Math.sqrt(num); i++) {
            if (num % i == 0) {
                return false;
            }
        }
        return true;
    }
}
