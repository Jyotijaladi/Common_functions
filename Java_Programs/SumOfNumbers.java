import java.util.Scanner;
public class SumOfNumbers {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Please provide numbers as command line arguments");
            return;
        }

        int sum = 0;

        for (String arg : args) {
            sum += Integer.parseInt(arg);
        }

        System.out.println("Sum: " + sum);
    }
}
