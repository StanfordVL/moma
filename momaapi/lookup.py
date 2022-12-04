import itertools
import json
import os
import os.path as osp
import pickle
import shutil

from .data import Bidict, LazyDict, Metadatum, Act, SAct, HOI, Clip

"""
The following functions are publicly available:
 - retrieve()
 - map_id()
 - map_cid()

retrieve(): accesses the value given a key
 - split -> ids_act (one-to-many): retrieve(kind='id_act', key=split)
 - id_act -> ann_act, metadatum (one-to-one): retrieve(kind='ann_act' or 'metadatum', key=id_act)
 - id_sact -> ann_sact (one-to-one): retrieve(kind='ann_sact', key=id_sact)
 - id_hoi -> ann_hoi, clip (one-to-one): retrieve(kind='ann_hoi' or 'clip', key=id_hoi)

map_id(): maps instance IDs across the MOMA hierarchy
 - id_act -> ids_sact (one-to-many): map_id(id_act=id_act, kind='sact')
 - id_act -> ids_hoi (one-to-many): map_id(id_act=id_act, kind='hoi')
 - id_sact -> id_act (one-to-one): map_id(id_sact=id_sact, kind='act')
 - id_sact -> ids_hoi (one-to-many): map_id(id_sact=id_sact, kind='hoi')
 - id_hoi -> id_sact (one-to-one): map_id(id_hoi=id_hoi, kind='sact')
 - id_hoi -> id_act (one-to-one): map_id(id_hoi=id_hoi, kind='act')
 
map_cid(): maps activity and sub-activity class IDs between few-shot and standard paradigms
 - cid_fs -> cid_std: map_cid(split=split, cid_act=cid_fs or cid_sact=cid_fs)
 - cid_std -> cid_fs: map_cid(split=split, cid_act=cid_std or cid_sact=cid_std)
"""


class Lookup:
    """
    Lookup utility class to help lookup annotations.
    """

    def __init__(self, dir_moma, taxonomy, reset_cache):
        self.taxonomy = taxonomy

        names = [
            "id_act_to_metadatum",
            "id_act_to_ann_act",
            "id_sact_to_ann_sact",
            "id_hoi_to_ann_hoi",
            "id_hoi_to_clip",
            "id_sact_to_id_act",
            "id_hoi_to_id_sact",
        ]
        names_lazy = ["id_sact_to_ann_sact", "id_hoi_to_ann_hoi", "id_hoi_to_clip"]
        names_bidict = ["id_sact_to_id_act", "id_hoi_to_id_sact"]
        self._read_anns(dir_moma, reset_cache, names, names_lazy, names_bidict)
        self.paradigm_and_split_to_ids_act = self._read_paradigms_and_splits(dir_moma)

    @staticmethod
    def _save_cache(dir_moma, data, names, names_lazy):
        dir_lookup = osp.join(dir_moma, "anns/cache/lookup")
        os.makedirs(dir_lookup, exist_ok=True)

        for name in names:
            if name in names_lazy:
                src, trg = name.split("_to_")
                os.makedirs(osp.join(dir_lookup, src), exist_ok=True)
                for key, value in data[name].items():
                    with open(
                        osp.join(dir_lookup, f'{name.replace("_to_", "/")}_{key}'), "wb"
                    ) as f:
                        pickle.dump(value, f)
            else:
                with open(osp.join(dir_lookup, name), "wb") as f:
                    pickle.dump(data[name], f)

    @staticmethod
    def _load_cache(dir_moma, names, names_lazy):
        dir_lookup = osp.join(dir_moma, "anns/cache/lookup")

        data = {}
        for name in names:
            if name in names_lazy:
                src, trg = name.split("_to_")
                data[name] = LazyDict(osp.join(dir_lookup, src), trg)
            else:
                with open(osp.join(dir_lookup, name), "rb") as f:
                    data[name] = pickle.load(f)

        return data

    def _read_anns(self, dir_moma, reset_cache, names, names_lazy, names_bidict):
        dir_lookup = osp.join(dir_moma, "anns/cache/lookup")
        if reset_cache and osp.exists(dir_lookup):
            shutil.rmtree(dir_lookup)

        try:
            data = self._load_cache(dir_moma, names, names_lazy)

        except FileNotFoundError:
            print("Compiling the Lookup class...")

            with open(osp.join(dir_moma, f"anns/anns.json"), "r") as f:
                anns_raw = json.load(f)

            if osp.exists(osp.join(dir_moma, f"videos/interaction_frames")):
                with open(
                    osp.join(dir_moma, f"videos/interaction_frames/timestamps.json"),
                    "r",
                ) as f:
                    info_clips = json.load(f)
            else:
                info_clips = None

            data = {name: {} for name in names}
            for ann_raw in anns_raw:
                ann_act_raw = ann_raw["activity"]
                data["id_act_to_metadatum"][ann_act_raw["id"]] = Metadatum(ann_raw)
                data["id_act_to_ann_act"][ann_act_raw["id"]] = Act(
                    ann_act_raw, self.taxonomy["act"]
                )
                scale_factor = data["id_act_to_metadatum"][
                    ann_act_raw["id"]
                ].scale_factor
                anns_sact_raw = ann_act_raw["sub_activities"]

                for ann_sact_raw in anns_sact_raw:
                    data["id_sact_to_ann_sact"][ann_sact_raw["id"]] = SAct(
                        ann_sact_raw,
                        scale_factor,
                        self.taxonomy["sact"],
                        self.taxonomy["actor"],
                        self.taxonomy["object"],
                        self.taxonomy["ia"],
                        self.taxonomy["ta"],
                        self.taxonomy["att"],
                        self.taxonomy["rel"],
                    )
                    data["id_sact_to_id_act"][ann_sact_raw["id"]] = ann_act_raw["id"]
                    anns_hoi_raw = ann_sact_raw["higher_order_interactions"]

                    for ann_hoi_raw in anns_hoi_raw:
                        data["id_hoi_to_ann_hoi"][ann_hoi_raw["id"]] = HOI(
                            ann_hoi_raw,
                            self.taxonomy["actor"],
                            self.taxonomy["object"],
                            self.taxonomy["ia"],
                            self.taxonomy["ta"],
                            self.taxonomy["att"],
                            self.taxonomy["rel"],
                        )
                        # Currently, only clips from the test set have been generated
                        if info_clips is not None and ann_hoi_raw["id"] in info_clips:
                            data["id_hoi_to_clip"][ann_hoi_raw["id"]] = Clip(
                                ann_hoi_raw, info_clips[ann_hoi_raw["id"]]
                            )
                        data["id_hoi_to_id_sact"][ann_hoi_raw["id"]] = ann_sact_raw[
                            "id"
                        ]

            self._save_cache(dir_moma, data, names, names_lazy)

        for name in names_bidict:
            data[name] = Bidict(data[name])
        for name in names:
            setattr(self, name, data[name])

    @staticmethod
    def _read_paradigms_and_splits(dir_moma):
        paradigms = ["standard", "few-shot"]
        splits = ["train", "val", "test"]

        paradigm_and_split_to_ids_act = {}
        for paradigm in paradigms:
            path_split = osp.join(
                dir_moma, f"anns/splits/{paradigm.replace('-', '_')}.json"
            )
            assert osp.isfile(
                path_split
            ), f"Dataset split file does not exist: {path_split}"
            with open(path_split, "r") as f:
                ids_act = json.load(f)
            for split in splits:
                paradigm_and_split_to_ids_act[f"{paradigm}_{split}"] = ids_act[split]

        return paradigm_and_split_to_ids_act

    def retrieve(self, kind, key=None):
        """
        Accesses the value given a key. There are several different ways to retrieve:

            * Convert a ``split`` into ``ids_act`` (one-to-many):
                ``retrieve(kind='id_act', key=split)``
            * Convert an ``id_act`` into an ``ann_act``, metadatum (one-to-one):
                ``retrieve(kind='ann_act' or 'metadatum', key=id_act)``
            * Convert an ``id_sact`` into an ``ann_sact`` (one-to-one):
                ``retrieve(kind='ann_sact', key=id_sact)``
            * Convert an ``id_hoi`` into an ``ann_hoi`` or a ``clip`` (one-to-one):
                ``retrieve(kind='ann_hoi' or 'clip', key=id_hoi)``

        :param kind: indicates the type of retrieval that is used
        :type kind: Literal["paradigms","splits","ids_act","ids_sact","ids_hoi","anns_act","metadata","anns_sact","anns_hoi","clips",]
        """
        if key is None:
            assert kind in [
                "paradigms",
                "splits",
                "ids_act",
                "ids_sact",
                "ids_hoi",
                "anns_act",
                "metadata",
                "anns_sact",
                "anns_hoi",
                "clips",
            ]

            if kind == "paradigms":
                return [
                    x.split("_")[0] for x in self.paradigm_and_split_to_ids_act.keys()
                ]
            elif kind == "splits":
                return [
                    x.split("_")[1] for x in self.paradigm_and_split_to_ids_act.keys()
                ]
            elif kind == "ids_act":
                return self.id_act_to_ann_act.keys()
            elif kind == "ids_sact":
                return self.id_sact_to_ann_sact.keys()
            elif kind == "ids_hoi":
                return self.id_hoi_to_ann_hoi.keys()
            elif kind == "anns_act":
                return self.id_act_to_ann_act.values()
            elif kind == "metadata":
                return self.id_act_to_metadatum.values()
            elif kind == "anns_sact":
                return self.id_sact_to_ann_sact.values()
            elif kind == "anns_hoi":
                return self.id_hoi_to_ann_hoi.values()
            elif kind == "clips":
                return self.id_hoi_to_clip.values()

        else:
            assert kind in [
                "ids_act",
                "ann_act",
                "metadatum",
                "ann_sact",
                "ann_hoi",
                "clip",
            ]

            if kind == "ids_act":
                return self.paradigm_and_split_to_ids_act[key]
            elif kind == "ann_act":
                return self.id_act_to_ann_act[key]
            elif kind == "metadatum":
                return self.id_act_to_metadatum[key]
            elif kind == "ann_sact":
                return self.id_sact_to_ann_sact[key]
            elif kind == "ann_hoi":
                return self.id_hoi_to_ann_hoi[key]
            elif kind == "clip":
                return self.id_hoi_to_clip[key]

        raise ValueError(f"retrieve(kind={kind}, key={key})")

    def map_id(self, kind, id_act=None, id_sact=None, id_hoi=None):
        """
        Maps instance IDs across the MOMA hierarchy. Usage:

            * Convert an ``id_act`` into ``ids_sact`` (one-to-many):
                ``map_id(id_act=id_act, kind='sact')``
            * Convert an ``id_act`` into ``ids_hoi`` (one-to-many):
                ``map_id(id_act=id_act, kind='hoi')``
            * Convert an ``id_sact`` into ``id_act`` (one-to-one):
                ``map_id(id_sact=id_sact, kind='act')``
            * Convert an ``id_sact`` into ``ids_hoi`` (one-to-many):
                ``map_id(id_sact=id_sact, kind='hoi')``
            * Convert an ``id_hoi`` into ``id_sact`` (one-to-one):
                ``map_id(id_hoi=id_hoi, kind='sact')``
            * Convert an ``id_hoi`` into ``id_act`` (one-to-one):
                ``map_id(id_hoi=id_hoi, kind='act')``

        """

        assert sum([x is not None for x in [id_act, id_sact, id_hoi]]) == 1
        assert kind in ["id_act", "id_sact", "ids_sact", "id_hoi", "ids_hoi"]

        if id_hoi is not None:
            assert kind in ["id_act", "id_sact"]

            if kind == "id_sact":
                id_sact = self.id_hoi_to_id_sact[id_hoi]
                return id_sact
            elif kind == "id_act":
                id_sact = self.id_hoi_to_id_sact[id_hoi]
                id_act = self.id_sact_to_id_act[id_sact]
                return id_act

        elif id_sact is not None:
            assert kind in ["id_act", "ids_hoi"]

            if kind == "id_act":
                id_act = self.id_sact_to_id_act[id_sact]
                return id_act
            elif kind == "ids_hoi":
                ids_hoi = self.id_hoi_to_id_sact.inverse[id_sact]
                return ids_hoi

        elif id_act is not None:
            assert kind in ["ids_sact", "ids_hoi"]

            if kind == "ids_sact":
                ids_sact = self.id_sact_to_id_act.inverse[id_act]
                return ids_sact
            elif kind == "ids_hoi":
                ids_hoi = itertools.chain(
                    *[
                        self.id_hoi_to_id_sact.inverse[id_sact]
                        for id_sact in self.id_sact_to_id_act.inverse[id_act]
                    ]
                )
                return ids_hoi

        raise ValueError

    def map_cid(self, paradigm, split=None, cid_act=None, cid_sact=None):
        assert sum([x is not None for x in [cid_act, cid_sact]]) == 1
        if cid_act is not None:
            kind = "act"
            cid_src = cid_act
        elif cid_sact is not None:
            kind = "sact"
            cid_src = cid_sact
        else:
            raise ValueError

        if paradigm == "standard":
            assert split is not None
            cname = self.taxonomy["few_shot"][kind][split][cid_src]
            cid_trg = self.taxonomy[kind].index(cname)

        elif paradigm == "few-shot":
            cname = self.taxonomy[kind][cid_src]
            split = self.taxonomy["few_shot"][kind].inverse[cname]
            cid_trg = self.taxonomy["few_shot"][kind][split].index(cname)

        else:
            raise ValueError

        return cid_trg
