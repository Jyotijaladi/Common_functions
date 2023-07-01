import java.util.Scanner;

public class Factorial {
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

public class RhombusPattern {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the limit: ");
        int limit = scanner.nextInt();

        for (int i = 1; i <= limit; i++) {
            for (int j = 1; j <= limit - i; j++) {
                System.out.print(" ");
            }

            for (int j = 1; j <= i; j++) {
                System.out.print("* ");
            }

            System.out.println();
        }

        for (int i = limit - 1; i >= 1; i--) {
            for (int j = 1; j <= limit - i; j++) {
                System.out.print(" ");
            }

            for (int j = 1; j <= i; j++) {
                System.out.print("* ");
            }

            System.out.println();
        }
    }
}

public class StringOperations {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();

        System.out.println("Length: " + str.length());
        System.out.println("Uppercase: " + str.toUpperCase());
        System.out.println("Lowercase: " + str.toLowerCase());
        System.out.println("Reverse: " + new StringBuilder(str).reverse());
    }
}

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

public class DistanceBetweenPoints {
    public static void main(String[] args) {
        if (args.length != 4) {
            System.out.println("Please provide four numbers (x1, y1, x2, y2) as command line arguments");
            return;
        }

        double x1 = Double.parseDouble(args[0]);
        double y1 = Double.parseDouble(args[1]);
        double x2 = Double.parseDouble(args[2]);
        double y2 = Double.parseDouble(args[3]);

        double distance = Math.sqrt(Math.pow((x2 - x1), 2) + Math.pow((y2 - y1), 2));
        System.out.println("Distance: " + distance);
    }
}

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

public class MouseEvents {
    // Implement your mouse events code here
}
