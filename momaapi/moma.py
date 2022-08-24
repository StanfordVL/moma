import itertools
import os.path as osp

from .taxonomy import Taxonomy
from .lookup import Lookup
from .statistics import Statistics
from typing_extensions import Literal


"""
The following functions are defined:
 - get_cids(): Get the class ID of a kind ('act', 'sact', etc.) that satisfies certain conditions
 - map_cids(): Map class IDs between standard class IDs and split-specific contiguous class IDs
 - get_cnames(): Given class IDs, return their class names
 - is_sact(): Check whether a certain time in an activity has a sub-activity
 - get_ids_act(): Get the unique activity instance IDs that satisfy certain conditions
 - get_ids_sact(): Get the unique sub-activity instance IDs that satisfy certain conditions
 - get_ids_hoi(): Get the unique higher-order interaction instance IDs that satisfy certain conditions
 - get_metadata(): Given activity instance IDs, return the metadata of the associated raw videos
 - get_anns_act(): Given activity instance IDs, return their annotations
 - get_anns_sact(): Given sub-activity instance IDs, return their annotations
 - get_anns_hoi(): Given higher-order interaction instance IDs, return their annotations
 - get_clip(): Given higher-order interaction instance IDs, return their clips
 - get_paths(): Given instance IDs, return data paths
 - sort(): Given a list of sub-activity or higher-order interaction instance IDs, return them in sorted order

The following paradigms are defined:
 - 'standard': Different splits share the same sets of activity classes and sub-activity classes
 - 'few-shot': Different splits have non-overlapping activity classes and sub-activity classes

The following attributes are defined:
 - statistics: an object that stores dataset statistics; please see statistics.py:95 for details
 - taxonomy: an object that stores dataset taxonomy; please see taxonomy.py:53 for details
 - num_classes: number of activity and sub-activity classes

 
Definitions:
 - kind: ['act', 'sact', 'hoi', 'actor', 'object', 'ia', 'ta', 'att', 'rel']
"""


class MOMA:
    """
    Class to interface with the MOMA-LRG dataset. Initialization requires passing in
    a directory containing the MOMA-LRG dataset.

    The MOMA object can be used for few-shot experiments, which reduces the number of
    classes and examples, or used in the standard paradigm.

    The following conventions are used throughout the documentation as shorthand:

    * ``act``: activity
    * ``sact``: sub-activity
    * ``hoi``: higher-order interaction
    * ``entity``: entity
    * ``ia``: intransitive action
    * ``ta``: transitive action
    * ``att``: attribute
    * ``rel``: relationship
    * ``ann``: annotation
    * ``id``: instance ID
    * ``cname``: class name
    * ``cid``: class ID

    :param dir_moma: directory containing the MOMA dataset
    :type dir_moma: str
    :param paradigm: the experiment configuration, which is either ``'standard'`` or ``'few-shot'``
    :type paradigm: Literal['standard', 'few-shot']
    :param reset_cache: flag that indicates whether to reset cached data
    :type reset_cache: bool
    :param taxonomy: a Taxonomy object containing information about the dataset taxonomy
    :type taxonomy: Taxonomy
    :param lookup: a Lookup object containing information about class IDs and class names
    :type lookup: Lookup
    :param statistics: a Statistics object that can generate dataset-level statics
    :param num_classes: the number of classes contained in the MOMA object
    :type num_classes: int
    """

    def __init__(
        self,
        dir_moma: str,
        paradigm: Literal["standard", "few-shot"] = "standard",
        reset_cache: bool = False,
    ):
        """
        Constructor for MOMA-LRG
        """
        assert osp.isdir(osp.join(dir_moma, "anns"))

        self.dir_moma = dir_moma
        self.paradigm = paradigm

        self.taxonomy = Taxonomy(dir_moma)
        self.lookup = Lookup(dir_moma, self.taxonomy, reset_cache)
        self.statistics = Statistics(dir_moma, self.taxonomy, self.lookup, reset_cache)

    @property
    def num_classes(self):
        return self.taxonomy.get_num_classes()[self.paradigm]

    def get_cids(
        self,
        kind: Literal["act", "sact", "actor", "object", "ia", "ta", "att", "rel"],
        threshold: int,
        split: Literal["train", "val", "test", "either", "all", "combined"],
    ) -> list:
        """
        :param kind: the kind of annotations needed to be retrieved
        :type kind: Literal['act', 'sact', 'actor', 'object', 'ia', 'ta', 'att', 'rel']
        :param threshold: exclude classes with fewer than this number of total instances
        :type threshold: int
        :param split: the split to be used for the retrieval. Here, ``train`` refers to
          the training set, ``val`` refers to the validation set, and ``test`` refers
          to the test set. ``either`` will exclude a class if the smallest number of
          instances in across splits is less than the threshold, `all` will exclude
          a class if the largest number of instances in across splits is less than the
          threshold, and ``combined`` will exclude a class if the smallest number of
          instances in across splits is less than the threshold
        :type split: Literal['train', 'val', 'test', 'either', 'all', 'combined']
        :return: a list of class IDs
        :rtype: List[int]
        """
        cids = self.statistics.get_cids(kind, threshold, self.paradigm, split)
        return cids

    def map_cids(
        self,
        split: Literal["train", "val", "test", "either", "all", "combined"],
        cids_act_contiguous: list = None,
        cids_act: list = None,
        cids_sact_contiguous: list = None,
        cids_sact: list = None,
    ) -> list:
        """
        Map class IDs between standard class IDs and split-specific contiguous class IDs.
        **For the few-shot paradigm only**.

        :param split: the dataset split to use
        :type split: Literal['train', 'val', 'test', 'either', 'all', 'combined']
        :param cids_act_contiguous: a list of contiguous class IDs in the activity set
        :type cids_act_contiguous: Optional[List[int]]
        :param cids_act: a list of class IDs in the activity set
        :type cids_act: Optional[List[int]]
        :param cids_sact_contiguous: a list of contiguous class IDs in the sub-activity set
        :type cids_sact_contiguous: Optional[List[int]]
        :param cids_sact: a list of class IDs in the sub-activity set
        :type cids_sact: Optional[List[int]]
        :return: mapping between standard class IDs and split-specific contiguous IDs
        """
        assert self.paradigm == "few-shot"
        assert (
            sum(
                [
                    x is not None
                    for x in [
                        cids_act_contiguous,
                        cids_act,
                        cids_sact_contiguous,
                        cids_sact,
                    ]
                ]
            )
            == 1
        )

        if cids_act_contiguous is not None:
            return [
                self.lookup.map_cid(paradigm="standard", split=split, cid_act=x)
                for x in cids_act_contiguous
            ]
        elif cids_act is not None:
            return [
                self.lookup.map_cid(paradigm="few-shot", split=split, cid_act=x)
                for x in cids_act
            ]
        elif cids_sact_contiguous is not None:
            return [
                self.lookup.map_cid(paradigm="standard", split=split, cid_sact=x)
                for x in cids_sact_contiguous
            ]
        elif cids_sact is not None:
            return [
                self.lookup.map_cid(paradigm="few-shot", split=split, cid_sact=x)
                for x in cids_sact
            ]
        else:
            raise ValueError

    def get_cnames(
        self,
        cids_act: list = None,
        cids_sact: list = None,
        cids_actor: list = None,
        cids_object: list = None,
        cids_ia: list = None,
        cids_ta: list = None,
        cids_att: list = None,
        cids_rel: list = None,
    ) -> list:
        """
        Returns the associated class names given the class IDs.

        :param cids_act: a list of class IDs of activities
        :type cids_act: Optional[List[int]]
        :param cids_sact: a list of class IDs of sub-activities
        :type cids_sact: Optional[List[int]]
        :param cids_actor: a list of class IDs of actors
        :type cids_actor: Optional[List[int]]
        :param cids_object: a list of class IDs of objects
        :type cids_object: Optional[List[int]]
        :param cids_ia: a list of class IDs of intransitive actions
        :type cids_ia: Optional[List[int]]
        :param cids_ta: a list of class IDs of transitive actions
        :type cids_ta: Optional[List[int]]
        :param cids_att: a list of class IDs of attributes
        :type cids_att: Optional[List[int]]
        :param cids_rel: a list of class IDs of relationships
        :type cids_rel: Optional[List[int]]
        :return: a list of class names
        :rtype: List[str]
        """
        args = [
            cids_act,
            cids_sact,
            cids_actor,
            cids_object,
            cids_ia,
            cids_ta,
            cids_att,
            cids_rel,
        ]
        kinds = ["act", "sact", "actor", "object", "ia", "ta", "att", "rel"]

        indices = [i for i, x in enumerate(args) if x is not None]
        assert len(indices) == 1

        cids = args[indices[0]]
        kind = kinds[indices[0]]

        cnames = [self.taxonomy[kind][cid] for cid in cids]
        return cnames

    def is_sact(self, id_act: int, time: int, absolute: bool = False) -> bool:
        """
        Checks whether a certain time in an activity has a sub-activity.

        :param id_act: activity ID
        :type id_act: int
        :param time: time in the activity
        :type time: int
        :param absolute: relative to the full video if ``True`` or relative to the
          activity video if ``False``
        :type absolute: bool
        """
        if not absolute:
            ann_act = self.lookup.retrieve("ann_act", id_act)
            time = ann_act.start + time

        is_sact = False
        ids_sact = self.lookup.map_id("ids_sact", id_act=id_act)
        for id_sact in ids_sact:
            ann_sact = self.lookup.retrieve("ann_sact", id_sact)
            if ann_sact.start <= time < ann_sact.end:
                is_sact = True

        return is_sact

    def get_ids_act(
        self,
        split: str = None,
        cnames_act: list = None,
        ids_sact: list = None,
        ids_hoi: list = None,
    ) -> list:
        """
        Get the unique activity instance IDs that satisfy certain conditions

        :param split: get activity IDs that belong to the given dataset split
        :type split: ``Union['train', 'val', 'test', 'either', 'all', 'combined']``
        :param cnames_act: get activity IDs that belong to the given activity classes
        :type cnames_act: list
        :param ids_sact: get activity IDs for given sub-activity IDs
        :type ids_sact: list
        :param ids_hoi: get activity IDs for given higher-order interaction IDs [ids_hoi]
        :type ids_hoi: list
        :return: a list of activity IDs
        :rtype: list
        """
        if all(x is None for x in [split, cnames_act, ids_sact, ids_hoi]):
            return sorted(self.lookup.retrieve("ids_act"))

        ids_act_intersection = []

        # split
        if split is not None:
            assert split in self.lookup.retrieve("splits")
            ids_act_intersection.append(
                self.lookup.retrieve("ids_act", f"{self.paradigm}_{split}")
            )

        # cnames_act
        if cnames_act is not None:
            ids_act = []
            for id_act in self.lookup.retrieve("ids_act"):
                ann_act = self.lookup.retrieve("ann_act", id_act)
                if ann_act.cname in cnames_act:
                    ids_act.append(id_act)
            ids_act_intersection.append(ids_act)

        # ids_sact
        if ids_sact is not None:
            ids_act = [
                self.lookup.map_id("id_act", id_sact=id_sact) for id_sact in ids_sact
            ]
            ids_act_intersection.append(ids_act)

        # ids_hoi
        if ids_hoi is not None:
            ids_act = [
                self.lookup.map_id("id_act", id_hoi=id_hoi) for id_hoi in ids_hoi
            ]
            ids_act_intersection.append(ids_act)

        ids_act_intersection = sorted(set.intersection(*map(set, ids_act_intersection)))
        return ids_act_intersection

    def get_ids_sact(
        self,
        split: str = None,
        cnames_sact: list = None,
        ids_act: list = None,
        ids_hoi: list = None,
        cnames_actor: list = None,
        cnames_object: list = None,
        cnames_ia: list = None,
        cnames_ta: list = None,
        cnames_att: list = None,
        cnames_rel: list = None,
    ) -> list:
        """
        Get the unique sub-activity instance IDs that satisfy certain conditions
        dataset split

        :param split: get sub-activity IDs [ids_sact] that belong to the given dataset split
        :type split: ``Union['train', 'val', 'test', 'either', 'all', 'combined']``
        :param cnames_sact: get sub-activity IDs [ids_sact] for given sub-activity class names [cnames_sact]
        :type cnames_sact: list
        :param ids_act: get sub-activity IDs [ids_sact] for given activity IDs [ids_act]
        :type ids_act: list
        :param ids_hoi: get sub-activity IDs [ids_sact] for given higher-order interaction IDs [ids_hoi]
        :type ids_hoi: list
        :param cnames_actor: get sub-activity IDs [ids_sact] for given actor class names [cnames_actor]
        :type cnames_actor: list
        :param cnames_object: get sub-activity IDs [ids_sact] for given object class names [cnames_object]
        :type cnames_object: list
        :param cnames_ia: get sub-activity IDs [ids_sact] for given intransitive action class names [cnames_ia]
        :type cnames_ia: list
        :param cnames_ta: get sub-activity IDs [ids_sact] for given transitive action class names [cnames_ta]
        :type cnames_ta: list
        :param cnames_att: get sub-activity IDs [ids_sact] for given attribute class names [cnames_att]
        :type cnames_att: list
        :param cnames_rel: get sub-activity IDs [ids_sact] for given relationship class names [cnames_rel]
        :type cnames_rel: list
        :return: a list of sub-activity IDs
        :rtype: list
        """
        if all(
            x is None
            for x in [
                split,
                cnames_sact,
                ids_act,
                ids_hoi,
                cnames_actor,
                cnames_object,
                cnames_ia,
                cnames_ta,
                cnames_att,
                cnames_rel,
            ]
        ):
            return sorted(self.lookup.retrieve("ids_sact"))

        ids_sact_intersection = []

        # split
        if split is not None:
            assert split in self.lookup.retrieve("splits")
            ids_sact = self.get_ids_sact(
                ids_act=self.lookup.retrieve("ids_act", f"{self.paradigm}_{split}")
            )
            ids_sact_intersection.append(ids_sact)

        # cnames_sact
        if cnames_sact is not None:
            ids_sact = []
            for id_sact in self.lookup.retrieve("ids_sact"):
                ann_sact = self.lookup.retrieve("ann_sact", id_sact)
                if ann_sact.cname in cnames_sact:
                    ids_sact.append(id_sact)
            ids_sact_intersection.append(ids_sact)

        # ids_act
        if ids_act is not None:
            ids_sact = itertools.chain(
                *[self.lookup.map_id("ids_sact", id_act=id_act) for id_act in ids_act]
            )
            ids_sact_intersection.append(ids_sact)

        # ids_hoi
        if ids_hoi is not None:
            ids_sact = [
                self.lookup.map_id("id_sact", id_hoi=id_hoi) for id_hoi in ids_hoi
            ]
            ids_sact_intersection.append(ids_sact)

        # cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel
        if not all(
            x is None
            for x in [
                cnames_actor,
                cnames_object,
                cnames_ia,
                cnames_ta,
                cnames_att,
                cnames_rel,
            ]
        ):
            kwargs = {
                "cnames_actor": cnames_actor,
                "cnames_object": cnames_object,
                "cnames_ia": cnames_ia,
                "cnames_ta": cnames_ta,
                "cnames_att": cnames_att,
                "cnames_rel": cnames_rel,
            }
            ids_sact = [
                self.lookup.map_id("id_sact", id_hoi=id_hoi)
                for id_hoi in self.get_ids_hoi(**kwargs)
            ]
            ids_sact_intersection.append(ids_sact)

        ids_sact_intersection = sorted(
            set.intersection(*map(set, ids_sact_intersection))
        )
        return ids_sact_intersection

    def get_ids_hoi(
        self,
        split: str = None,
        ids_act: list = None,
        ids_sact: list = None,
        cnames_actor: list = None,
        cnames_object: list = None,
        cnames_ia: list = None,
        cnames_ta: list = None,
        cnames_att: list = None,
        cnames_rel: list = None,
    ) -> list:
        """
        Get the unique higher-order interaction instance IDs that satisfy certain conditions
        dataset split

        :param split: get higher-order interaction IDs [ids_hoi] that belong to the given dataset split
        :type split: ``Union['train', 'val', 'test', 'either', 'all', 'combined']``
        :param ids_act: get higher-order interaction IDs [ids_hoi] for given activity IDs [ids_act]
        :type ids_act: list
        :param ids_sact: get higher-order interaction IDs [ids_hoi] for given sub-activity IDs [ids_sact]
        :type ids_sact: list
        :param cnames_actor: get higher-order interaction IDs [ids_hoi] for given actor class names [cnames_actor]
        :type cnames_actor: list
        :param cnames_object: get higher-order interaction IDs [ids_hoi] for given object class names [cnames_object]
        :type cnames_object: list
        :param cnames_ia: get higher-order interaction IDs [ids_hoi] for given intransitive action class names [cnames_ia]
        :type cnames_ia: list
        :param cnames_ta: get higher-order interaction IDs [ids_hoi] for given transitive action class names [cnames_ta]
        :type cnames_ta: list
        :param cnames_att: get higher-order interaction IDs [ids_hoi] for given attribute class names [cnames_att]
        :type cnames_att: list
        :param cnames_rel: get higher-order interaction IDs [ids_hoi] for given relationship class names [cnames_rel]
        :type cnames_rel: list
        """
        if all(
            x is None
            for x in [
                split,
                ids_act,
                ids_sact,
                cnames_actor,
                cnames_object,
                cnames_ia,
                cnames_ta,
                cnames_att,
                cnames_rel,
            ]
        ):
            return sorted(self.lookup.retrieve("ids_hoi"))

        ids_hoi_intersection = []

        # split
        if split is not None:
            assert split in self.lookup.retrieve("splits")
            ids_hoi = self.get_ids_hoi(
                ids_act=self.lookup.retrieve("ids_act", f"{self.paradigm}_{split}")
            )
            ids_hoi_intersection.append(ids_hoi)

        # ids_act
        if ids_act is not None:
            ids_hoi = itertools.chain(
                *[self.lookup.map_id("ids_hoi", id_act=id_act) for id_act in ids_act]
            )
            ids_hoi_intersection.append(ids_hoi)

        # ids_sact
        if ids_sact is not None:
            ids_hoi = itertools.chain(
                *[
                    self.lookup.map_id("ids_hoi", id_sact=id_sact)
                    for id_sact in ids_sact
                ]
            )
            ids_hoi_intersection.append(ids_hoi)

        # cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel
        cnames_dict = {
            "actors": cnames_actor,
            "objects": cnames_object,
            "ias": cnames_ia,
            "tas": cnames_ta,
            "atts": cnames_att,
            "rels": cnames_rel,
        }
        for var, cnames in cnames_dict.items():
            if cnames is not None:
                ids_hoi = []
                for id_hoi in self.lookup.retrieve("ids_hoi"):
                    ann_hoi = self.lookup.retrieve("ann_hoi", id_hoi)
                    if not set(cnames).isdisjoint(
                        [x.cname for x in getattr(ann_hoi, var)]
                    ):
                        ids_hoi.append(id_hoi)
                ids_hoi_intersection.append(ids_hoi)

        ids_hoi_intersection = sorted(set.intersection(*map(set, ids_hoi_intersection)))
        return ids_hoi_intersection

    def get_metadata(self, ids_act: list) -> list:
        """
        Get the metadata for the given activity IDs. The metadata returned
        is that associated with the raw videos that contain instances of the
        activity IDs.

        :param ids_act: get metadata for the given activity IDs
        :return: video metadata for the given activity ID
        :rtype: list
        """
        return [self.lookup.retrieve("metadatum", id_act) for id_act in ids_act]

    def get_anns_act(self, ids_act: list) -> list:
        """
        Given activity instance IDs, return their annotations

        :param ids_act: activity instance IDs
        :return: annotations for the given activity instance IDs
        :rtype: list
        """
        return [self.lookup.retrieve("ann_act", id_act) for id_act in ids_act]

    def get_anns_sact(self, ids_sact: list) -> list:
        """
        Given sub-activity instance IDs, return their annotations

        :param ids_sact: sub-activity instance IDs
        :return: annotations for the given sub-activity instance IDs
        :rtype: list
        """
        return [self.lookup.retrieve("ann_sact", id_sact) for id_sact in ids_sact]

    def get_anns_hoi(self, ids_hoi: list) -> list:
        """
        Given higher-order interaction instance IDs, return their annotations

        :param ids_hoi: higher-order interaction instance IDs
        :return: annotations for the given higher-order interaction instance IDs
        :rtype: list
        """
        return [self.lookup.retrieve("ann_hoi", id_hoi) for id_hoi in ids_hoi]

    def get_clips(self, ids_hoi: list) -> list:
        """
        Given higher-order interaction instance IDs, return their clips

        :param ids_hoi: higher-order interaction instance IDs
        :return: clips for the given higher-order interaction instance IDs
        :rtype: list
        """
        return [self.lookup.retrieve("clip", id_hoi) for id_hoi in ids_hoi]

    def get_paths(
        self,
        ids_act: list = None,
        ids_sact: list = None,
        ids_hoi: list = None,
        id_hoi_clip: str = None,
        full_res: bool = False,
        sanity_check: bool = True,
    ) -> list:
        """
        Given activity, sub-activity, higher-order interaction, or clip IDs, return the paths to the videos.

        :param ids_act: activity instance IDs
        :type ids_act: list
        :param ids_sact: sub-activity instance IDs
        :type ids_sact: list
        :param ids_hoi: higher-order interaction instance IDs
        :type ids_hoi: list
        :param id_hoi_clip: clip ID
        :type id_hoi_clip: str
        :param full_res: return full-resolution videos
        :type full_res: bool
        :param sanity_check: check that the video exists
        :type sanity_check: bool
        :return: paths to the videos
        :rtype: list
        """
        assert (
            sum([x is not None for x in [ids_act, ids_sact, ids_hoi, id_hoi_clip]]) == 1
        )

        if ids_act is not None:
            paths = [
                osp.join(
                    self.dir_moma,
                    f"videos/activity{'_fr' if full_res else ''}/{id_act}.mp4",
                )
                for id_act in ids_act
            ]
        elif ids_sact is not None:
            paths = [
                osp.join(
                    self.dir_moma,
                    f"videos/sub_activity{'_fr' if full_res else ''}/{id_sact}.mp4",
                )
                for id_sact in ids_sact
            ]
        elif ids_hoi is not None:
            paths = [
                osp.join(self.dir_moma, f"videos/interaction/{id_hoi}.jpg")
                for id_hoi in ids_hoi
            ]
        else:
            assert id_hoi_clip is not None
            clip = self.get_clips(ids_hoi=[id_hoi_clip])[0]
            times = [x[1] for x in clip.neighbors] + [clip.time]
            paths = [
                osp.join(self.dir_moma, f"videos/interaction_frames/{x[0]}.jpg")
                for x in clip.neighbors
            ] + [osp.join(self.dir_moma, f"videos/interaction/{id_hoi_clip}.jpg")]
            paths = [x for _, x in sorted(zip(times, paths))]

        if sanity_check and not all(osp.exists(path) for path in paths):
            paths_missing = [path for path in paths if not osp.exists(path)]
            paths_missing = (
                paths_missing[:5] if len(paths_missing) > 5 else paths_missing
            )
            assert False, f"{len(paths_missing)} paths do not exist: {paths_missing}"

        return paths

    def sort(
        self, ids_sact: list = None, ids_hoi: list = None, sanity_check: bool = True
    ):
        """
        Given a list of sub-activity or higher-order interaction instance IDs, return them in sorted order
        by when they occured in the video.

        :param ids_sact: sub-activity instance IDs
        :type ids_sact: list
        :param ids_hoi: higher-order interaction instance IDs
        :type ids_hoi: list
        :param sanity_check: check that the video exists
        :type sanity_check: bool
        :return: sorted IDs
        :rtype: list
        """
        assert sum([x is not None for x in [ids_sact, ids_hoi]]) == 1

        if ids_sact is not None:
            if sanity_check:  # make sure they come from the same activity instance
                id_act = self.get_ids_act(ids_sact=[ids_sact[0]])[0]
                ids_sact_all = self.get_ids_sact(ids_act=[id_act])
                assert set(ids_sact).issubset(set(ids_sact_all))
            ids_sact = sorted(
                ids_sact, key=lambda x: self.get_anns_sact(ids_sact=[x])[0].start
            )
            return ids_sact
        else:
            if sanity_check:  # make sure they come from the same sub-activity instance
                id_sact = self.get_ids_sact(ids_hoi=[ids_hoi[0]])[0]
                ids_hoi_all = self.get_ids_hoi(ids_sact=[id_sact])
                assert set(ids_hoi).issubset(set(ids_hoi_all))
            ids_hoi = sorted(
                ids_hoi, key=lambda x: self.get_anns_hoi(ids_hoi=[x])[0].time
            )
            return ids_hoi
