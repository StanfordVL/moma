import argparse

from momaapi import MOMA


def verify_api(args):
    moma = MOMA(args.dir_moma)


def verify_dataset():
    # TODO: verify dataset layout
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir-moma", type=str, default="/home/alan/data/moma-lrg"
    )
    args = parser.parse_args()

    verify_api(args)
    verify_dataset()

    print("Dataset and API are verified.")


if __name__ == "__main__":
    main()
