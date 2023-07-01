import java.util.Scanner;
public class AverageOfNumbers {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Please provide numbers as command line arguments");
            return;
        }

        double sum = 0;
        int count = args.length;

        for (String arg : args) {
            sum += Double.parseDouble(arg);
        }

        double average = sum / count;
        System.out.println("Average: " + average);
    }
}
