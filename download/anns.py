import argparse
import gdown
import os
import os.path as osp
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir-moma", type=str, default="/home/alanzluo/data/moma-lrg"
    )
    args = parser.parse_args()

    url = "https://drive.google.com/uc?id=1stizUmyHY6aNxxbxUPD5DvoibBvUrKZW"
    fname = "anns.tar.xz"

    gdown.download(url, osp.join(args.dir_moma, fname), quiet=False)

    file = tarfile.open(osp.join(args.dir_moma, fname))
    file.extractall(args.dir_moma)
    file.close()
    os.remove(osp.join(args.dir_moma, fname))


if __name__ == "__main__":
    main()
