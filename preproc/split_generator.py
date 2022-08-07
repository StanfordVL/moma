from collections import defaultdict
import itertools
import json
import os
import random


class SplitGenerator:
    def __init__(self, dir_moma):
        self.dir_moma = dir_moma

        with open(os.path.join(self.dir_moma, "anns/anns.json"), "r") as f:
            anns = json.load(f)

        self.ids_act = [ann["activity"]["id"] for ann in anns]
        self.cnames_act = [ann["activity"]["class_name"] for ann in anns]
        self.cnames_sact = [
            [sact["class_name"] for sact in ann["activity"]["sub_activities"]]
            for ann in anns
        ]

        self.cname_to_ids = defaultdict(list)
        for id_act, cname_act in zip(self.ids_act, self.cnames_act):
            self.cname_to_ids[cname_act].append(id_act)

    def generate_standard_splits(self, ratio):
        size_train = round(len(self.ids_act) * ratio)
        size_val = round(size_train * (1 - ratio))
        size_train = size_train - size_val

        for i in range(10000):
            print(f"Generating standard splits: {i+1}th attempt")
            indices = random.sample(range(len(self.ids_act)), len(self.ids_act))
            indices_train = indices[:size_train]
            indices_val = indices[size_train : (size_train + size_val)]
            indices_test = indices[(size_train + size_val) :]

            cnames_act_train = set([self.cnames_act[j] for j in indices_train])
            cnames_act_val = set([self.cnames_act[j] for j in indices_val])
            cnames_act_test = set([self.cnames_act[j] for j in indices_test])

            cnames_sact_train = set(
                itertools.chain(*[self.cnames_sact[j] for j in indices_train])
            )
            cnames_sact_val = set(
                itertools.chain(*[self.cnames_sact[j] for j in indices_val])
            )
            cnames_sact_test = set(
                itertools.chain(*[self.cnames_sact[j] for j in indices_test])
            )

            if len(cnames_act_train) == len(cnames_act_val) == len(
                cnames_act_test
            ) and len(cnames_sact_train) == len(cnames_sact_val) == len(
                cnames_sact_test
            ):
                break

        ids_act_train = [self.ids_act[j] for j in indices_train]
        ids_act_val = [self.ids_act[j] for j in indices_val]
        ids_act_test = [self.ids_act[j] for j in indices_test]
        path_split = os.path.join(self.dir_moma, "anns/split_std.json")
        with open(path_split, "w") as f:
            json.dump(
                {"train": ids_act_train, "val": ids_act_val, "test": ids_act_test},
                f,
                indent=2,
                sort_keys=False,
            )

    def generate_few_shot_splits(self):
        with open(os.path.join(self.dir_moma, "anns/taxonomy/few_shot.json"), "r") as f:
            split_to_cnames = json.load(f)

        # need ids_act given cnames
        path_split = os.path.join(self.dir_moma, "anns/split_fs.json")
        output = {}
        for split, cnames in split_to_cnames.items():
            output[split] = itertools.chain.from_iterable(
                [self.cname_to_ids[cname] for cname in split_to_cnames[split]]
            )
        with open(path_split, "w") as f:
            json.dump(output, f, indent=2, sort_keys=False)
