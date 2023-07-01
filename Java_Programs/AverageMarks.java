import java.util.Scanner;
public class AverageMarks {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Please provide marks as command line arguments");
            return;
        }

        int total = 0;
        int count = args.length;

        for (String arg : args) {
            total += Integer.parseInt(arg);
        }

        double average = (double) total / count;
        System.out.println("Average marks: " + average);
    }
}
