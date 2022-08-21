import argparse
from multiprocessing import Pool

from momaapi import MOMA, AnnVisualizer


def save_gif(id_sact):
    visualizer.show_sact(id_sact, vstack=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dir-moma", type=str, default="/media/hdd/moma-lrg")
    parser.add_argument("-o", "--dir-vis", type=str, default="/media/hdd/moma-lrg/vis")
    parser.add_argument("-n", "--num-cpus", type=int, default=8)
    args = parser.parse_args()

    moma = MOMA(args.dir_moma)

    global visualizer
    visualizer = AnnVisualizer(moma, args.dir_vis)
    ids_sact = moma.get_ids_sact()

    with Pool(args.num_cpus) as p:
        p.map(save_gif, ids_sact)


if __name__ == "__main__":
    main()
