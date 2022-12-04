import itertools
import json
import jsbeautifier
import numpy as np
import os
import os.path as osp


class Statistics(dict):
    def __init__(self, dir_moma, taxonomy, lookup, reset_cache):
        super().__init__()
        self._taxonomy = taxonomy
        self._lookup = lookup
        self.statistics = self._read_statistics(dir_moma, reset_cache)
        self._sanity_check()

    def get_cids(self, kind, threshold, paradigm, split):
        assert paradigm in self._lookup.retrieve("paradigms")
        assert split in self._lookup.retrieve("splits") + ["either", "all", "combined"]

        # exclude a class if the smallest number of instances in across splits is less than the threshold
        if split == "either":
            distribution = np.stack(
                [
                    self.statistics[f"{paradigm}_{_split}"][kind]["distribution"]
                    for _split in self._lookup.retrieve("splits")
                ]
            )
            distribution = np.amin(distribution, axis=0)

        # exclude a class if the largest number of instances in across splits is less than the threshold
        elif split == "all":
            distribution = np.stack(
                [
                    self.statistics[f"{paradigm}_{_split}"][kind]["distribution"]
                    for _split in self._lookup.retrieve("splits")
                ]
            )
            distribution = np.amax(distribution, axis=0)

        # exclude a class if the number of instances in the entire dataset is less than the threshold
        elif split == "combined":
            distribution = np.array(self.statistics["all"][kind]["distribution"])

        else:
            distribution = np.array(
                self.statistics[f"{paradigm}_{split}"][kind]["distribution"]
            )

        cids = np.where(distribution >= threshold)[0].tolist()
        return cids

    def _sanity_check(self):
        # standard
        assert (
            self.statistics["all"]["act"]["num_classes"]
            == len(self._taxonomy["act"])
            == self.statistics["standard_train"]["act"]["num_classes"]
            == self.statistics["standard_val"]["act"]["num_classes"]
            == self.statistics["standard_test"]["act"]["num_classes"]
        )
        assert (
            self.statistics["all"]["sact"]["num_classes"]
            == len(self._taxonomy["sact"])
            == self.statistics["standard_train"]["sact"]["num_classes"]
            == self.statistics["standard_val"]["sact"]["num_classes"]
            == self.statistics["standard_test"]["sact"]["num_classes"]
        )
        assert (
            self.statistics["all"]["actor"]["num_classes"]
            == len(self._taxonomy["actor"])
            == self.statistics["standard_train"]["actor"]["num_classes"]
            == self.statistics["standard_val"]["actor"]["num_classes"]
            == self.statistics["standard_test"]["actor"]["num_classes"]
        )
        # TODO: fix object taxonomy
        # assert self.statistics['all']['object']['num_classes'] == len(self._taxonomy['object']) == \
        #        self.statistics['standard_train']['object']['num_classes'] == \
        #        self.statistics['standard_val']['object']['num_classes'] == \
        #        self.statistics['standard_test']['object']['num_classes']

        # few-shot
        assert self.statistics["few-shot_train"]["act"][
            "num_classes"
        ] + self.statistics["few-shot_val"]["act"]["num_classes"] + self.statistics[
            "few-shot_test"
        ][
            "act"
        ][
            "num_classes"
        ] == len(
            self._taxonomy["act"]
        )
        assert self.statistics["few-shot_train"]["sact"][
            "num_classes"
        ] + self.statistics["few-shot_val"]["sact"]["num_classes"] + self.statistics[
            "few-shot_test"
        ][
            "sact"
        ][
            "num_classes"
        ] == len(
            self._taxonomy["sact"]
        )

    def _save_cache(self, path_statistics, statistics):
        with open(path_statistics, "w") as f:
            options = jsbeautifier.default_options()
            options.indent_size = 4
            f.write(jsbeautifier.beautify(json.dumps(statistics), options))

    def _load_cache(self, path_statistics):
        with open(path_statistics, "r") as f:
            statistics = json.load(f)
        return statistics

    def _read_statistics(self, dir_moma, reset_cache):
        paradigms = self._lookup.retrieve("paradigms")
        splits = self._lookup.retrieve("splits")

        path_statistics = osp.join(dir_moma, "anns/cache/statistics.json")
        if reset_cache and osp.exists(path_statistics):
            os.remove(path_statistics)

        if osp.exists(path_statistics):
            statistics = self._load_cache(path_statistics)

        else:
            print("Compiling the Statistics class...")
            statistics = {"all": self._get_statistics()}
            for paradigm, split in itertools.product(paradigms, splits):
                statistics[f"{paradigm}_{split}"] = self._get_statistics(
                    paradigm, split
                )
            self._save_cache(path_statistics, statistics)

        return statistics

    @staticmethod
    def _get_duration(anns):
        duration_total = sum(ann.end - ann.start for ann in anns)
        duration_avg = duration_total / len(anns)
        duration_min = min(ann.end - ann.start for ann in anns)
        duration_max = max(ann.end - ann.start for ann in anns)
        return duration_total, duration_avg, duration_min, duration_max

    def _get_statistics(self, paradigm=None, split=None):
        # subsample metadata, anns_act, anns_sact, and anns_hoi
        if paradigm is None and split is None:
            metadata = self._lookup.retrieve("metadata")
            anns_act = self._lookup.retrieve("anns_act")
            anns_sact = self._lookup.retrieve("anns_sact")
            anns_hoi = self._lookup.retrieve("anns_hoi")
        else:
            assert paradigm is not None and split is not None
            ids_act = self._lookup.retrieve("ids_act", f"{paradigm}_{split}")
            metadata = [
                self._lookup.retrieve("metadatum", id_act) for id_act in ids_act
            ]
            anns_act = [self._lookup.retrieve("ann_act", id_act) for id_act in ids_act]
            ids_sact = list(
                itertools.chain(
                    *[
                        self._lookup.map_id("ids_sact", id_act=id_act)
                        for id_act in ids_act
                    ]
                )
            )
            anns_sact = [
                self._lookup.retrieve("ann_sact", id_sact) for id_sact in ids_sact
            ]
            ids_hoi = list(
                itertools.chain(
                    *[
                        self._lookup.map_id("ids_hoi", id_sact=id_sact)
                        for id_sact in ids_sact
                    ]
                )
            )
            anns_hoi = [self._lookup.retrieve("ann_hoi", id_hoi) for id_hoi in ids_hoi]

        # number of classes and instances
        num_acts = len(anns_act)
        num_classes_act = len(set([ann_act.cid for ann_act in anns_act]))
        num_sacts = len(anns_sact)
        num_classes_sact = len(set([ann_sact.cid for ann_sact in anns_sact]))
        num_hois = len(anns_hoi)

        num_actors_image = sum([len(ann_hoi.actors) for ann_hoi in anns_hoi])
        num_actors_video = sum([len(ann_sact.ids_actor) for ann_sact in anns_sact])
        num_classes_actor = len(
            set([actor.cid for ann_hoi in anns_hoi for actor in ann_hoi.actors])
        )
        num_objects_image = sum([len(ann_hoi.objects) for ann_hoi in anns_hoi])
        num_objects_video = sum([len(ann_sact.ids_object) for ann_sact in anns_sact])
        num_classes_object = len(
            set([object.cid for ann_hoi in anns_hoi for object in ann_hoi.objects])
        )

        num_ias = sum([len(ann_hoi.ias) for ann_hoi in anns_hoi])
        num_classes_ia = len(
            set([ia.cid for ann_hoi in anns_hoi for ia in ann_hoi.ias])
        )
        num_tas = sum([len(ann_hoi.tas) for ann_hoi in anns_hoi])
        num_classes_ta = len(
            set([ta.cid for ann_hoi in anns_hoi for ta in ann_hoi.tas])
        )
        num_atts = sum([len(ann_hoi.atts) for ann_hoi in anns_hoi])
        num_classes_att = len(
            set([att.cid for ann_hoi in anns_hoi for att in ann_hoi.atts])
        )
        num_rels = sum([len(ann_hoi.rels) for ann_hoi in anns_hoi])
        num_classes_rel = len(
            set([rel.cid for ann_hoi in anns_hoi for rel in ann_hoi.rels])
        )

        # durations
        duration_total_raw = sum(metadatum.duration for metadatum in metadata)
        (
            duration_total_act,
            duration_avg_act,
            duration_min_act,
            duration_max_act,
        ) = self._get_duration(anns_act)
        (
            duration_total_sact,
            duration_avg_sact,
            duration_min_sact,
            duration_max_sact,
        ) = self._get_duration(anns_sact)

        # class distributions
        bincount_act = np.bincount(
            [ann_act.cid for ann_act in anns_act], minlength=len(self._taxonomy["act"])
        ).tolist()
        bincount_sact = np.bincount(
            [ann_sact.cid for ann_sact in anns_sact],
            minlength=len(self._taxonomy["sact"]),
        ).tolist()
        (
            bincount_actor,
            bincount_object,
            bincount_ia,
            bincount_ta,
            bincount_att,
            bincount_rel,
        ) = ([], [], [], [], [], [])
        for ann_hoi in anns_hoi:
            bincount_actor += [actor.cid for actor in ann_hoi.actors]
            bincount_object += [object.cid for object in ann_hoi.objects]
            bincount_ia += [ia.cid for ia in ann_hoi.ias]
            bincount_ta += [ta.cid for ta in ann_hoi.tas]
            bincount_att += [att.cid for att in ann_hoi.atts]
            bincount_rel += [rel.cid for rel in ann_hoi.rels]
        bincount_actor = np.bincount(
            bincount_actor, minlength=len(self._taxonomy["actor"])
        ).tolist()
        bincount_object = np.bincount(
            bincount_object, minlength=len(self._taxonomy["object"])
        ).tolist()
        bincount_ia = np.bincount(
            bincount_ia, minlength=len(self._taxonomy["ia"])
        ).tolist()
        bincount_ta = np.bincount(
            bincount_ta, minlength=len(self._taxonomy["ta"])
        ).tolist()
        bincount_att = np.bincount(
            bincount_att, minlength=len(self._taxonomy["att"])
        ).tolist()
        bincount_rel = np.bincount(
            bincount_rel, minlength=len(self._taxonomy["rel"])
        ).tolist()

        # curate statistics
        statistics = {
            "raw": {"duration_total": duration_total_raw},
            "act": {
                "num_instances": num_acts,
                "num_classes": num_classes_act,
                "duration_avg": duration_avg_act,
                "duration_min": duration_min_act,
                "duration_max": duration_max_act,
                "duration_total": duration_total_act,
                "distribution": bincount_act,
            },
            "sact": {
                "num_instances": num_sacts,
                "num_classes": num_classes_sact,
                "duration_avg": duration_avg_sact,
                "duration_min": duration_min_sact,
                "duration_max": duration_max_sact,
                "duration_total": duration_total_sact,
                "distribution": bincount_sact,
            },
            "hoi": {"num_instances": num_hois,},
            "actor": {
                "num_instances_image": num_actors_image,
                "num_instances_video": num_actors_video,
                "num_classes": num_classes_actor,
                "distribution": bincount_actor,
            },
            "object": {
                "num_instances_image": num_objects_image,
                "num_instances_video": num_objects_video,
                "num_classes": num_classes_object,
                "distribution": bincount_object,
            },
            "ia": {
                "num_instances": num_ias,
                "num_classes": num_classes_ia,
                "distribution": bincount_ia,
            },
            "ta": {
                "num_instances": num_tas,
                "num_classes": num_classes_ta,
                "distribution": bincount_ta,
            },
            "att": {
                "num_instances": num_atts,
                "num_classes": num_classes_att,
                "distribution": bincount_att,
            },
            "rel": {
                "num_instances": num_rels,
                "num_classes": num_classes_rel,
                "distribution": bincount_rel,
            },
        }

        return statistics

    def keys(self):
        return self.statistics.keys()

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    def __getitem__(self, key):
        return self.statistics[key]

    def __len__(self):
        return len(self.statistics.keys())

    def __repr__(self):
        return repr(self.statistics)
