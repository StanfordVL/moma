import csv
import os
from pathlib import Path

from momaapi import MOMA


def main():
    dir_moma = os.path.join(Path.home(), "data/moma")
    dir_out = os.path.join(Path.home(), "data/moma/third_party/slowfast")
    moma = MOMA(dir_moma)

    os.makedirs(os.path.join(dir_out, "act"), exist_ok=True)
    os.makedirs(os.path.join(dir_out, "sact"), exist_ok=True)

    ids_act_train = moma.get_ids_act(split="train")
    anns_act_train = moma.get_anns_act(ids_act=ids_act_train)
    labels_train = [anns_act_train.cid for anns_act_train in anns_act_train]
    paths_train = moma.get_paths(ids_act=ids_act_train)
    with open(os.path.join(dir_out, "act/train.csv"), "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for path_train, label_train in zip(paths_train, labels_train):
            writer.writerow([path_train, label_train])
        f.close()

    ids_act_val = moma.get_ids_act(split="val")
    anns_act_val = moma.get_anns_act(ids_act=ids_act_val)
    labels_val = [anns_act_val.cid for anns_act_val in anns_act_val]
    paths_val = moma.get_paths(ids_act=ids_act_val)
    with open(os.path.join(dir_out, "act/val.csv"), "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for path_val, label_val in zip(paths_val, labels_val):
            writer.writerow([path_val, label_val])
        f.close()

    ids_sact_train = moma.get_ids_sact(split="train")
    anns_sact_train = moma.get_anns_sact(ids_sact=ids_sact_train)
    labels_train = [anns_sact_train.cid for anns_sact_train in anns_sact_train]
    paths_train = moma.get_paths(ids_sact=ids_sact_train)
    with open(os.path.join(dir_out, "sact/train.csv"), "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for path_train, label_train in zip(paths_train, labels_train):
            writer.writerow([path_train, label_train])
        f.close()

    ids_sact_val = moma.get_ids_sact(split="val")
    anns_sact_val = moma.get_anns_sact(ids_sact=ids_sact_val)
    labels_val = [anns_sact_val.cid for anns_sact_val in anns_sact_val]
    paths_val = moma.get_paths(ids_sact=ids_sact_val)
    with open(os.path.join(dir_out, "sact/val.csv"), "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for path_val, label_val in zip(paths_val, labels_val):
            writer.writerow([path_val, label_val])
        f.close()


if __name__ == "__main__":
    main()
