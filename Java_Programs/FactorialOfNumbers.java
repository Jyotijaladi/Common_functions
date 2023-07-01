package pack;
import java.util.Scanner;

public class FactorialOfNumbers {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Please provide numbers as command line arguments");
            return;
        }

        for (String arg : args) {
            int num = Integer.parseInt(arg);
            int factorial = 1;

            for (int i = 1; i <= num; i++) {
                factorial *= i;
            }

            System.out.println("Factorial of " + num + " is: " + factorial);
        }
    }
}
