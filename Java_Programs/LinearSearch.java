import java.util.Scanner;
public class LinearSearch {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Please provide numbers as command line arguments");
            return;
        }

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the element to search: ");
        int target = scanner.nextInt();

        boolean found = false;

        for (String arg : args) {
            int num = Integer.parseInt(arg);

            if (num == target) {
                found = true;
                break;
            }
        }

        if (found) {
            System.out.println("Element found");
        } else {
            System.out.println("Element not found");
        }
    }
}
