import argparse

from momaapi import MOMA, AnnVisualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir-moma", type=str, default="/media/hdd/moma-lrg")
    parser.add_argument("-d", "--dir-vis", type=str, default="/media/hdd/moma-lrg/vis")
    args = parser.parse_args()

    moma = MOMA(args.dir_moma)

    visualizer = AnnVisualizer(moma, args.dir_vis)
    ids_sact = moma.get_ids_sact()
    for id_sact in ids_sact:
        visualizer.show_sact(id_sact, vstack=False)


if __name__ == "__main__":
    main()
