import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='This is a model for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # run(args.input, args.output)