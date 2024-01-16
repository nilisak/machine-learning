import argparse
import math


def fibonacci_sequence(length):
    if length == 1:
        sequence = [0]
    elif length == 2:
        sequence = [0, 1]
    else:
        sequence = [0, 1]
        while len(sequence) < length:
            sequence.append(sequence[-1] + sequence[-2])
    return sequence


def prime_sequence(length):
    sequence = []
    num = 2
    while len(sequence) < length:
        if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
            sequence.append(num)
        num += 1
    return sequence


def square_sequence(length):
    sequence = [i**2 for i in range(1, length + 1)]
    return sequence


def triangular_sequence(length):
    sequence = []
    current_sum = 0
    for i in range(1, length + 1):
        current_sum += i
        sequence.append(current_sum)
    return sequence


def factorial_sequence(length):
    sequence = []
    for i in range(1, length + 1):
        if i == 1:
            sequence.append(1)
        else:
            sequence.append(sequence[i - 2] * i)

    return sequence


def main(args):
    if args.sequence == "fibonacci":
        return fibonacci_sequence(args.length)
    elif args.sequence == "prime":
        return prime_sequence(args.length)
    elif args.sequence == "square":
        return square_sequence(args.length)
    elif args.sequence == "triangular":
        return triangular_sequence(args.length)
    elif args.sequence == "factorial":
        return factorial_sequence(args.length)
    else:
        raise ValueError(
            "Invalid sequence name. Supported sequences: fibonacci, prime, square, triangular, factorial"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mathematical sequences iteratively.")
    parser.add_argument("--length", type=int, help="Length of the computed sequence", required=True)
    parser.add_argument(
        "--sequence",
        type=str,
        help="Name of the sequence",
        choices=["fibonacci", "prime", "square", "triangular", "factorial"],
        required=True,
    )
    args = parser.parse_args()
    result_sequence = main(args)
    print(result_sequence)
