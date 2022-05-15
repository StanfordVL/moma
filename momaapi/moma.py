import itertools
import numpy as np
import os

from .io import *


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
     - full_res: load full-resolution videos
     - few_shot: load few-shot splits
     - load_val: load the validation set separately
    """
    assert os.path.isdir(os.path.join(dir_moma, 'anns')) and os.path.isdir(os.path.join(dir_moma, 'videos'))

    self.dir_moma = dir_moma
    self.full_res = full_res
    self.load_val = load_val

    io = IO(dir_moma)
    self.taxonomy, self.taxonomy_fs, self.lvis_mapper = io.read_taxonomy()
    self.metadata, self.id_act_to_ann_act, self.id_sact_to_ann_sact, self.id_hoi_to_ann_hoi, \
        self.id_sact_to_id_act, self.id_hoi_to_id_sact, self.windows = io.read_anns()
    self.split_to_ids_act = io.read_splits(few_shot, load_val)

    assert set(self.get_ids_act()) == set(itertools.chain.from_iterable(self.split_to_ids_act.values()))

    self.statistics, self.distributions = self.__get_summaries()

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
      distribution = np.stack([self.distributions[split][concept] for split in self.split_to_ids_act], axis=0)
      distribution = np.amin(distribution, axis=0).tolist()
    elif split == 'both':  # exclude if < threshold in all splits
      distribution = np.stack([self.distributions[split][concept] for split in self.split_to_ids_act], axis=0)
      distribution = np.amax(distribution, axis=0).tolist()
    else:
      distribution = self.distributions[split][concept]

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
        for id_hoi, ann_hoi in self.id_hoi_to_ann_hoi.items():
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
    window = self.windows[id_hoi]
    now = self.get_anns_hoi(ids_hoi=[id_hoi])[0].time
    window = [[os.path.join(self.dir_moma, f'videos/interaction_frames/{fname}.jpg'), time] for fname, time in window]+ \
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

  def __get_summaries(self):
    statistics_all, distributions_all = self.__get_summary()
    statistics = {'all': statistics_all}
    distributions = {'all': distributions_all}

    for split in self.split_to_ids_act:
      statistics_split, distributions_split = self.__get_summary(split)
      statistics[split] = statistics_split
      distributions[split] = distributions_split

    return statistics, distributions

  def __get_summary(self, split=None):
    if split is None:
      metadata = self.metadata.values()
      anns_act = self.id_act_to_ann_act.values()
      anns_sact = self.id_sact_to_ann_sact.values()
      anns_hoi = self.id_hoi_to_ann_hoi.values()
    else:
      assert split in self.split_to_ids_act
      ids_act = self.split_to_ids_act[split]
      metadata = self.get_metadata(ids_act=ids_act)
      anns_act = self.get_anns_act(ids_act=ids_act)
      ids_sact = self.get_ids_sact(ids_act=ids_act)
      anns_sact = self.get_anns_sact(ids_sact=ids_sact)
      ids_hoi = self.get_ids_hoi(ids_act=ids_act)
      anns_hoi = self.get_anns_hoi(ids_hoi=ids_hoi)
  
    num_acts = len(anns_act)
    num_classes_act = len(self.taxonomy['act'])
    num_sacts = len(anns_sact)
    num_classes_sact = len(self.taxonomy['sact'])
    num_hois = len(anns_hoi)
  
    num_actors_image = sum([len(ann_hoi.actors) for ann_hoi in anns_hoi])
    num_actors_video = sum([len(ann_sact.ids_actor) for ann_sact in anns_sact])
    num_classes_actor = len(self.taxonomy['actor'])
    num_objects_image = sum([len(ann_hoi.objects) for ann_hoi in anns_hoi])
    num_objects_video = sum([len(ann_sact.ids_object) for ann_sact in anns_sact])
    num_classes_object = len(self.taxonomy['object'])
  
    num_ias = sum([len(ann_hoi.ias) for ann_hoi in anns_hoi])
    num_classes_ia = len(self.taxonomy['ia'])
    num_tas = sum([len(ann_hoi.tas) for ann_hoi in anns_hoi])
    num_classes_ta = len(self.taxonomy['ta'])
    num_atts = sum([len(ann_hoi.atts) for ann_hoi in anns_hoi])
    num_classes_att = len(self.taxonomy['att'])
    num_rels = sum([len(ann_hoi.rels) for ann_hoi in anns_hoi])
    num_classes_rel = len(self.taxonomy['rel'])
  
    duration_total_raw = sum(metadatum.duration for metadatum in metadata)
  
    duration_total_act = sum(ann_act.end-ann_act.start for ann_act in anns_act)
    duration_avg_act = duration_total_act/len(anns_act)
    duration_min_act = min(ann_act.end-ann_act.start for ann_act in anns_act)
    duration_max_act = max(ann_act.end-ann_act.start for ann_act in anns_act)
  
    duration_total_sact = sum(ann_sact.end-ann_sact.start for ann_sact in anns_sact)
    duration_avg_sact = duration_total_sact/len(anns_sact)
    duration_min_sact = min(ann_sact.end-ann_sact.start for ann_sact in anns_sact)
    duration_max_sact = max(ann_sact.end-ann_sact.start for ann_sact in anns_sact)
  
    bincount_act = np.bincount([ann_act.cid for ann_act in anns_act], minlength=num_classes_act).tolist()
    bincount_sact = np.bincount([ann_sact.cid for ann_sact in anns_sact], minlength=num_classes_sact).tolist()
    bincount_actor, bincount_object, bincount_ia, bincount_ta, bincount_att, bincount_rel = [], [], [], [], [], []
    for ann_hoi in anns_hoi:
      bincount_actor += [actor.cid for actor in ann_hoi.actors]
      bincount_object += [object.cid for object in ann_hoi.objects]
      bincount_ia += [ia.cid for ia in ann_hoi.ias]
      bincount_ta += [ta.cid for ta in ann_hoi.tas]
      bincount_att += [att.cid for att in ann_hoi.atts]
      bincount_rel += [rel.cid for rel in ann_hoi.rels]
    bincount_actor = np.bincount(bincount_actor, minlength=num_classes_actor).tolist()
    bincount_object = np.bincount(bincount_object, minlength=num_classes_object).tolist()
    bincount_ia = np.bincount(bincount_ia, minlength=num_classes_ia).tolist()
    bincount_ta = np.bincount(bincount_ta, minlength=num_classes_ta).tolist()
    bincount_att = np.bincount(bincount_att, minlength=num_classes_att).tolist()
    bincount_rel = np.bincount(bincount_rel, minlength=num_classes_rel).tolist()
  
    statistics = {
      'raw': {
        'duration_total': duration_total_raw
      },
      'act': {
        'num_instances': num_acts,
        'num_classes': num_classes_act,
        'duration_avg': duration_avg_act,
        'duration_min': duration_min_act,
        'duration_max': duration_max_act,
        'duration_total': duration_total_act
      },
      'sact': {
        'num_instances': num_sacts,
        'num_classes': num_classes_sact,
        'duration_avg': duration_avg_sact,
        'duration_min': duration_min_sact,
        'duration_max': duration_max_sact,
        'duration_total': duration_total_sact
      },
      'hoi': {
        'num_instances': num_hois,
      },
      'actor': {
        'num_instances_image': num_actors_image,
        'num_instances_video': num_actors_video,
        'num_classes': num_classes_actor
      },
      'object': {
        'num_instances_image': num_objects_image,
        'num_instances_video': num_objects_video,
        'num_classes': num_classes_object
      },
      'ia': {
        'num_instances': num_ias,
        'num_classes': num_classes_ia
      },
      'ta': {
        'num_instances': num_tas,
        'num_classes': num_classes_ta
      },
      'att': {
        'num_instances': num_atts,
        'num_classes': num_classes_att
      },
      'rel': {
        'num_instances': num_rels,
        'num_classes': num_classes_rel
      },
    }
  
    distributions = {
      'act': bincount_act,
      'sact': bincount_sact,
      'actor': bincount_actor,
      'object': bincount_object,
      'ia': bincount_ia,
      'ta': bincount_ta,
      'att': bincount_att,
      'rel': bincount_rel,
    }
  
    return statistics, distributions
