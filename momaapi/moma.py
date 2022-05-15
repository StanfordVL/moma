import itertools
import numpy as np
import os

from .io import IO
from .statistics import Statistics


"""
The following functions are defined:
 - get_taxonomy: Get the taxonomy of a concept ('act', 'sact', etc.)
 - get_cnames: Get the class name of a concept ('act', 'sact', etc.) that satisfy certain conditions
 - is_sact: Check whether a certain time in an activity is a sub-activity
 - get_ids_act: Get the unique activity instance IDs that satisfy certain conditions
 - get_ids_sact: Get the unique sub-activity instance IDs that satisfy certain conditions
 - get_ids_hoi: Get the unique higher-order interaction instance IDs that satisfy certain conditions
 - get_metadata: Given activity instance IDs, return the metadata of the associated raw videos
 - get_anns_act: Given activity instance IDs, return their annotations
 - get_anns_sact: Given sub-activity instance IDs, return their annotations
 - get_anns_hoi: Given higher-order interaction instance IDs, return their annotations
 - get_paths: Given instance IDs, return data paths
 - get_paths_window: Given an HOI instance ID, return window paths
 - sort: Given sub-activity or higher-order interaction instance IDs, return them in sorted order
 - get_cid_fs： Get the consecutive few-shot class id given a class id
 
The following attributes are defined:
 - statistics: TODO
 
Acronyms:
 - act: activity
 - sact: sub-activity
 - hoi: higher-order interaction
 - entity: entity
 - ia: intransitive action
 - ta: transitive action
 - att: attribute
 - rel: relationship
 - ann: annotation
 - id: instance ID
 - cname: class name
 - cid: class ID
 
Definitions:
 - concept: ['act', 'sact', 'hoi', 'actor', 'object', 'ia', 'ta', 'att', 'rel']
 - kind: for entity, ['actor', 'object']; 
         for predicate ['ia', 'ta', 'att', 'rel']
"""


class MOMA:
  def __init__(self,
               dir_moma: str,
               full_res: bool=False,
               few_shot: bool=False,
               load_val: bool=False):
    """
     - dir_moma: directory of the MOMA dataset
     - full_res: whether to load full-resolution videos
     - few_shot: whether to load few-shot splits
     - load_val: whether to load the validation set separately
    """
    assert os.path.isdir(os.path.join(dir_moma, 'anns')) and os.path.isdir(os.path.join(dir_moma, 'videos'))

    self.dir_moma = dir_moma
    self.full_res = full_res
    self.load_val = load_val

    io = IO(dir_moma)
    self.taxonomy, self.taxonomy_fs, self.lvis_mapper = io.read_taxonomy()
    self.metadata, self.id_act_to_ann_act, self.id_sact_to_ann_sact, self.id_hoi_to_ann_hoi, \
        self.id_sact_to_id_act, self.id_hoi_to_id_sact, self.id_hoi_to_window = io.read_anns()
    self.split_to_ids_act = io.read_splits(few_shot, load_val)

    assert set(self.get_ids_act()) == set(itertools.chain.from_iterable(self.split_to_ids_act.values()))

    self.statistics = Statistics(self)

  def get_taxonomy(self, concept):
    assert concept in self.taxonomy
    return self.taxonomy[concept]

  def get_cnames(self, concept, threshold=None, split=None):
    """
     - concept: currently only support 'actor' and 'object'
     - threshold: exclude classes with fewer than this number of instances
     - split: 'train', 'val', 'test', 'all', 'either'
    """
    assert concept in ['actor', 'object']

    if threshold is None:
      return self.get_taxonomy(concept)

    assert split is not None
    if split == 'either':  # exclude if < threshold in either one split
      distribution = np.stack([self.statistics[split][concept]['distribution'] for split in self.split_to_ids_act])
      distribution = np.amin(distribution, axis=0).tolist()
    elif split == 'both':  # exclude if < threshold in all splits
      distribution = np.stack([self.statistics[split][concept]['distribution'] for split in self.split_to_ids_act])
      distribution = np.amax(distribution, axis=0).tolist()
    else:
      distribution = self.statistics[split][concept]['distribution']

    cnames = []
    for i, cname in enumerate(self.get_taxonomy(concept)):
      if distribution[i] >= threshold:
        cnames.append(cname)

    return cnames

  def is_sact(self, id_act, time, absolute=False):
    """ Check whether a certain time in an activity is a sub-activity
     - id_act: activity ID
     - time: time in seconds
     - absolute: relative to the full video (True) or relative to the activity video (False)
    """
    if not absolute:
      ann_act = self.id_act_to_ann_act[id_act]
      time = ann_act.start+time

    is_sact = False
    ids_sact = self.id_sact_to_id_act.inverse[id_act]
    for id_sact in ids_sact:
      ann_sact = self.id_sact_to_ann_sact[id_sact]
      if ann_sact.start <= time < ann_sact.end:
        is_sact = True

    return is_sact

  def get_ids_act(self, split: str=None, cnames_act: list=None,
                  ids_sact: list=None, ids_hoi: list=None) -> list:

    """ Get the unique activity instance IDs that satisfy certain conditions
    dataset split
     - split: get activity IDs [ids_act] that belong to the given dataset split
    same-level
     - cnames_act: get activity IDs [ids_act] for given activity class names [cnames_act]
    bottom-up
     - ids_sact: get activity IDs [ids_act] for given sub-activity IDs [ids_sact]
     - ids_hoi: get activity IDs [ids_act] for given higher-order interaction IDs [ids_hoi]
    """
    if all(x is None for x in [split, cnames_act, ids_sact, ids_hoi]):
      return sorted(self.id_act_to_ann_act.keys())

    ids_act_intersection = []

    # split
    if split is not None:
      assert split in self.split_to_ids_act
      ids_act_intersection.append(self.split_to_ids_act[split])

    # cnames_act
    if cnames_act is not None:
      ids_act = []
      for id_act, ann_act in self.id_act_to_ann_act.items():
        if ann_act.cname in cnames_act:
          ids_act.append(id_act)
      ids_act_intersection.append(ids_act)

    # ids_sact
    if ids_sact is not None:
      ids_act = [self.id_sact_to_id_act[id_sact] for id_sact in ids_sact]
      ids_act_intersection.append(ids_act)

    # ids_hoi
    if ids_hoi is not None:
      ids_act = [self.id_sact_to_id_act[self.id_hoi_to_id_sact[id_hoi]] for id_hoi in ids_hoi]
      ids_act_intersection.append(ids_act)

    ids_act_intersection = sorted(set.intersection(*map(set, ids_act_intersection)))
    return ids_act_intersection

  def get_ids_sact(self, split: str=None,
                   cnames_sact: list=None, ids_act: list=None, ids_hoi: list=None,
                   cnames_actor: list=None, cnames_object: list=None,
                   cnames_ia: list=None, cnames_ta: list=None,
                   cnames_att: list=None, cnames_rel: list=None) -> list:

    """ Get the unique sub-activity instance IDs that satisfy certain conditions
    dataset split
     - split: get sub-activity IDs [ids_sact] that belong to the given dataset split
    same-level
     - cnames_sact: get sub-activity IDs [ids_sact] for given sub-activity class names [cnames_sact]
    top-down
     - ids_act: get sub-activity IDs [ids_sact] for given activity IDs [ids_act]
    bottom-up
     - ids_hoi: get sub-activity IDs [ids_sact] for given higher-order interaction IDs [ids_hoi]
     - cnames_actor: get sub-activity IDs [ids_sact] for given actor class names [cnames_actor]
     - cnames_object: get sub-activity IDs [ids_sact] for given object class names [cnames_object]
     - cnames_ia: get sub-activity IDs [ids_sact] for given intransitive action class names [cnames_ia]
     - cnames_ta: get sub-activity IDs [ids_sact] for given transitive action class names [cnames_ta]
     - cnames_att: get sub-activity IDs [ids_sact] for given attribute class names [cnames_att]
     - cnames_rel: get sub-activity IDs [ids_sact] for given relationship class names [cnames_rel]
    """
    if all(x is None for x in [split, cnames_sact, ids_act, ids_hoi, cnames_actor, cnames_object,
                               cnames_ia, cnames_ta, cnames_att, cnames_rel]):
      return sorted(self.id_sact_to_ann_sact.keys())

    ids_sact_intersection = []

    # split
    if split is not None:
      assert split in self.split_to_ids_act
      ids_sact = self.get_ids_sact(ids_act=self.split_to_ids_act[split])
      ids_sact_intersection.append(ids_sact)

    # cnames_sact
    if cnames_sact is not None:
      ids_sact = []
      for id_sact, ann_sact in self.id_sact_to_ann_sact.items():
        if ann_sact.cname in cnames_sact:
          ids_sact.append(id_sact)
      ids_sact_intersection.append(ids_sact)

    # ids_act
    if ids_act is not None:
      ids_sact = itertools.chain(*[self.id_sact_to_id_act.inverse[id_act] for id_act in ids_act])
      ids_sact_intersection.append(ids_sact)

    # ids_hoi
    if ids_hoi is not None:
      ids_sact = [self.id_hoi_to_id_sact[id_hoi] for id_hoi in ids_hoi]
      ids_sact_intersection.append(ids_sact)

    # cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel
    if not all(x is None for x in [cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel]):
      kwargs = {'cnames_actor': cnames_actor, 'cnames_object': cnames_object,
                'cnames_ia': cnames_ia, 'cnames_ta': cnames_ta,
                'cnames_att': cnames_att, 'cnames_rel': cnames_rel}
      ids_sact = [self.id_hoi_to_id_sact[id_hoi] for id_hoi in self.get_ids_hoi(**kwargs)]
      ids_sact_intersection.append(ids_sact)

    ids_sact_intersection = sorted(set.intersection(*map(set, ids_sact_intersection)))
    return ids_sact_intersection

  def get_ids_hoi(self, split: str=None,
                  ids_act: list=None, ids_sact: list=None,
                  cnames_actor: list=None, cnames_object: list=None,
                  cnames_ia: list=None, cnames_ta: list=None,
                  cnames_att: list=None, cnames_rel: list=None) -> list:
    """ Get the unique higher-order interaction instance IDs that satisfy certain conditions
    dataset split
     - split: get higher-order interaction IDs [ids_hoi] that belong to the given dataset split
    top-down
     - ids_act: get higher-order interaction IDs [ids_hoi] for given activity IDs [ids_act]
     - ids_sact: get higher-order interaction IDs [ids_hoi] for given sub-activity IDs [ids_sact]
    bottom-up
     - cnames_actor: get higher-order interaction IDs [ids_hoi] for given actor class names [cnames_actor]
     - cnames_object: get higher-order interaction IDs [ids_hoi] for given object class names [cnames_object]
     - cnames_ia: get higher-order interaction IDs [ids_hoi] for given intransitive action class names [cnames_ia]
     - cnames_ta: get higher-order interaction IDs [ids_hoi] for given transitive action class names [cnames_ta]
     - cnames_att: get higher-order interaction IDs [ids_hoi] for given attribute class names [cnames_att]
     - cnames_rel: get higher-order interaction IDs [ids_hoi] for given relationship class names [cnames_rel]
    """
    if all(x is None for x in [split, ids_act, ids_sact, cnames_actor, cnames_object,
                               cnames_ia, cnames_ta, cnames_att, cnames_rel]):
      return sorted(self.id_hoi_to_ann_hoi.keys())

    ids_hoi_intersection = []

    # split
    if split is not None:
      assert split in self.split_to_ids_act
      ids_hoi = self.get_ids_hoi(ids_act=self.split_to_ids_act[split])
      ids_hoi_intersection.append(ids_hoi)

    # ids_act
    if ids_act is not None:
      ids_hoi = itertools.chain(*[self.id_hoi_to_id_sact.inverse[id_sact]
                                  for id_act in ids_act
                                  for id_sact in self.id_sact_to_id_act.inverse[id_act]])
      ids_hoi_intersection.append(ids_hoi)

    # ids_sact
    if ids_sact is not None:
      ids_hoi = itertools.chain(*[self.id_hoi_to_id_sact.inverse[id_sact] for id_sact in ids_sact])
      ids_hoi_intersection.append(ids_hoi)

    # cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel
    cnames_dict = {'actors': cnames_actor, 'objects': cnames_object,
                   'ias': cnames_ia, 'tas': cnames_ta,
                   'atts': cnames_att, 'rels': cnames_rel}
    for var, cnames in cnames_dict.items():
      if cnames is not None:
        ids_hoi = []
        for id_hoi in self.id_hoi_to_ann_hoi.keys():
          ann_hoi = self.id_hoi_to_ann_hoi[id_hoi]
          if not set(cnames).isdisjoint([x.cname for x in getattr(ann_hoi, var)]):
            ids_hoi.append(id_hoi)
        ids_hoi_intersection.append(ids_hoi)

    ids_hoi_intersection = sorted(set.intersection(*map(set, ids_hoi_intersection)))
    return ids_hoi_intersection

  def get_metadata(self, ids_act: list) -> list:
    return [self.metadata[id_act] for id_act in ids_act]

  def get_anns_act(self, ids_act: list) -> list:
    return [self.id_act_to_ann_act[id_act] for id_act in ids_act]
  
  def get_anns_sact(self, ids_sact: list) -> list:
     return [self.id_sact_to_ann_sact[id_sact] for id_sact in ids_sact]
    
  def get_anns_hoi(self, ids_hoi: list) -> list:
     return [self.id_hoi_to_ann_hoi[id_hoi] for id_hoi in ids_hoi]

  def get_paths(self,
                ids_act: list=None,
                ids_sact: list=None,
                ids_hoi: list=None,
                sanity_check: bool=True) -> list:
    assert sum([x is not None for x in [ids_act, ids_sact, ids_hoi]]) == 1

    if ids_act is not None:
      paths = [os.path.join(self.dir_moma, f"videos/activity{'_fr' if self.full_res else ''}/{id_act}.mp4")
               for id_act in ids_act]
    elif ids_sact is not None:
      paths = [os.path.join(self.dir_moma, f"videos/sub_activity{'_fr' if self.full_res else ''}/{id_sact}.mp4")
               for id_sact in ids_sact]
    else:  # hoi
      paths = [os.path.join(self.dir_moma, f'videos/interaction/{id_hoi}.jpg') for id_hoi in ids_hoi]

    if sanity_check and not all(os.path.exists(path) for path in paths):
      paths_missing = [path for path in paths if not os.path.exists(path)]
      paths_missing = paths_missing[:5] if len(paths_missing) > 5 else paths_missing
      assert False, f'{len(paths_missing)} paths do not exist: {paths_missing}'
    return paths

  def get_paths_window(self, id_hoi):
    """ Given a higher-order interaction ID, return
     - a path to the 1s video clip centered at the higher-order interaction (<1s if exceeds the raw video boundary)
     - paths to 5 frames centered at the higher-order interaction (<5 frames if exceeds the raw video boundary)
    """
    window = self.id_hoi_to_window[id_hoi]
    now = self.get_anns_hoi(ids_hoi=[id_hoi])[0].time
    window = [[os.path.join(self.dir_moma, f'videos/interaction_frames/{fname}.jpg'), time] for fname, time in window]+\
             [[os.path.join(self.dir_moma, f'videos/interaction/{id_hoi}.jpg'), now]]
    window = sorted(window, key=lambda x: x[1])

    path_video = os.path.join(self.dir_moma, f'videos/interaction_video/{id_hoi}.mp4')
    paths_frame = [path_frame for path_frame, time in window]

    return path_video, paths_frame

  def sort(self, ids_sact: list=None, ids_hoi: list=None, sanity_check: bool=True):
    assert sum([x is not None for x in [ids_sact, ids_hoi]]) == 1

    if ids_sact is not None:
      if sanity_check:  # make sure they come from the same activity instance
        id_act = self.get_ids_act(ids_sact=[ids_sact[0]])[0]
        ids_sact_all = self.get_ids_sact(ids_act=[id_act])
        assert set(ids_sact).issubset(set(ids_sact_all))
      ids_sact = sorted(ids_sact, key=lambda x: self.get_anns_sact(ids_sact=[x])[0].start)
      return ids_sact
    else:
      if sanity_check:  # make sure they come from the same sub-activity instance
        id_sact = self.get_ids_sact(ids_hoi=[ids_hoi[0]])[0]
        ids_hoi_all = self.get_ids_hoi(ids_sact=[id_sact])
        assert set(ids_hoi).issubset(set(ids_hoi_all))
      ids_hoi = sorted(ids_hoi, key=lambda x: self.get_anns_hoi(ids_hoi=[x])[0].time)
      return ids_hoi

  def get_cid_fs(self, cid, concept, split):
    assert concept in ['act', 'sact']
    cname = self.taxonomy[concept][cid]
    if cname in self.taxonomy_fs[f'{concept}_val'] and not self.load_val:
      cid_fs = self.taxonomy_fs[f'{concept}_val'].index(cname)
      cid_fs += len(self.taxonomy_fs[f'{concept}_train'])
    else:
      cid_fs = self.taxonomy_fs[f'{concept}_{split}'].index(cname)
    return cid_fs
