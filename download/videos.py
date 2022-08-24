import argparse
import os.path as osp
from pytube import YouTube

from momaapi import MOMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir-moma", type=str, default="/home/alanzluo/data/moma-lrg"
    )
    args = parser.parse_args()

    moma = MOMA(args.dir_moma)

    ids_act = moma.get_ids_act()
    for id_act in ids_act:
        try:
            url = osp.join(f"https://www.youtube.com/watch?v={id_act}")
            yt = YouTube(url)
        except:
            print(f"Connection Error: {id_act}")

        # fmt: off
        yt.streams\
          .filter(progressive=True, file_extension="mp4")\
          .order_by("resolution")\
          .desc()\
          .first()\
          .download(osp.join(args.dir_moma, f"videos/raw/{id_act}.mp4"))
        # fmt: on


if __name__ == "__main__":
    main()
