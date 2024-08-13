import argparse
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    file_list = glob.glob(os.path.join(args.output_path, f"*.mae"))
    with open('paths_list.txt', 'w') as fh:
        fh.write('\n'.join(file_list))

if __name__ == "__main__":
    main()
