import itertools
import numpy as np
import os.path as osp

from .taxonomy import Taxonomy
from .lookup import Lookup
from .statistics import Statistics


"""
The following functions are defined:
 - get_cnames(): Get the class name of a kind ('act', 'sact', etc.) that satisfy certain conditions
 - is_sact(): Check whether a certain time in an activity has a sub-activity
 - get_ids_act(): Get the unique activity instance IDs that satisfy certain conditions
 - get_ids_sact(): Get the unique sub-activity instance IDs that satisfy certain conditions
 - get_ids_hoi(): Get the unique higher-order interaction instance IDs that satisfy certain conditions
 - get_metadata(): Given activity instance IDs, return the metadata of the associated raw videos
 - get_anns_act(): Given activity instance IDs, return their annotations
 - get_anns_sact(): Given sub-activity instance IDs, return their annotations
 - get_anns_hoi(): Given higher-order interaction instance IDs, return their annotations
 - get_paths(): Given instance IDs, return data paths
 - get_paths_window(): Given an HOI instance ID, return window paths
 - sort(): Given a list of sub-activity or higher-order interaction instance IDs, return them in sorted order

The following paradigms are defined:
 - 'standard': Different splits share the same sets of activity classes and sub-activity classes
 - 'few-shot': Different splits have non-overlapping activity classes and sub-activity classes

The following attributes are defined:
 - statistics: an object that stores dataset statistics; please see statistics.py:95 for details
 - taxonomy: an object that stores dataset taxonomy; please see taxonomy.py:53 for details
 - num_classes: number of activity and sub-activity classes
 
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
 - kind: ['act', 'sact', 'hoi', 'actor', 'object', 'ia', 'ta', 'att', 'rel']
"""


class MOMA:
  def __init__(self, dir_moma: str, paradigm: str='standard', load_val: bool=False, full_res: bool=False):
    """
     - dir_moma: directory of the MOMA dataset
     - paradigm: 'standard' or 'few-shot'
     - load_val: whether to load the validation set separately
     - full_res: whether to load full-resolution videos
    """
    assert osp.isdir(osp.join(dir_moma, 'anns')) and osp.isdir(osp.join(dir_moma, 'videos'))

    self.dir_moma = dir_moma
    self.paradigm = paradigm
    self.load_val = load_val
    self.full_res = full_res

    self.taxonomy = Taxonomy(dir_moma)
    self.lookup = Lookup(dir_moma, self.taxonomy, paradigm, load_val)
    self.statistics = Statistics(dir_moma, self.taxonomy, self.lookup)

  @property
  def num_classes(self):
    kinds = ['act', 'sact']
    if self.paradigm == 'standard':
      output = {kind: self.taxonomy.get_num_classes(self.paradigm, kind) for kind in kinds}

    elif self.paradigm == 'few-shot':
      output = {}
      for kind in kinds:
        if self.load_val:
          output[f'{kind}_train'] = self.taxonomy.get_num_classes(self.paradigm, kind, 'train')
          output[f'{kind}_val'] = self.taxonomy.get_num_classes(self.paradigm, kind, 'val')
        else:
          output[f'{kind}_train'] = self.taxonomy.get_num_classes(self.paradigm, kind, 'train')+\
                                    self.taxonomy.get_num_classes(self.paradigm, kind, 'val')
        output[f'{kind}_test'] = self.taxonomy.get_num_classes(self.paradigm, kind, 'test')

    else:
      raise ValueError

    return output

  def get_cnames(self, kind, threshold=None, split=None):
    """
     - kind: currently only support 'actor' and 'object'
     - threshold: exclude classes with fewer than this number of instances
     - split: 'train', 'val', 'test', 'all', 'either'
    """
    assert kind in ['actor', 'object']

    if threshold is None:
      return self.taxonomy[kind]

    assert split is not None
    if split == 'either':  # exclude if < threshold in either one split
      distribution = np.stack([self.statistics[split][kind]['distribution'] 
                               for split in self.lookup.retrieve('splits')])
      distribution = np.amin(distribution, axis=0).tolist()
    elif split == 'all':  # exclude if < threshold in all splits
      distribution = np.stack([self.statistics[split][kind]['distribution'] 
                               for split in self.lookup.retrieve('splits')])
      distribution = np.amax(distribution, axis=0).tolist()
    else:
      distribution = self.statistics[split][kind]['distribution']

    cnames = []
    for i, cname in enumerate(self.taxonomy[kind]):
      if distribution[i] >= threshold:
        cnames.append(cname)

    return cnames

  def is_sact(self, id_act, time, absolute=False):
    """ Check whether a certain time in an activity has a sub-activity
     - id_act: activity ID
     - time: time in seconds
     - absolute: relative to the full video (True) or relative to the activity video (False)
    """
    if not absolute:
      ann_act = self.lookup.retrieve('ann_act', id_act)
      time = ann_act.start+time

    is_sact = False
    ids_sact = self.lookup.trace('ids_sact', id_act=id_act)
    for id_sact in ids_sact:
      ann_sact = self.lookup.retrieve('ann_sact', id_sact)
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
      return sorted(self.lookup.retrieve('ids_act'))

    ids_act_intersection = []

    # split
    if split is not None:
      assert split in self.lookup.retrieve('splits')
      ids_act_intersection.append(self.lookup.retrieve('ids_act', split))

    # cnames_act
    if cnames_act is not None:
      ids_act = []
      for id_act in self.lookup.retrieve('ids_act'):
        ann_act = self.lookup.retrieve('ann_act', id_act)
        if ann_act.cname in cnames_act:
          ids_act.append(id_act)
      ids_act_intersection.append(ids_act)

    # ids_sact
    if ids_sact is not None:
      ids_act = [self.lookup.trace('id_act', id_sact=id_sact) for id_sact in ids_sact]
      ids_act_intersection.append(ids_act)

    # ids_hoi
    if ids_hoi is not None:
      ids_act = [self.lookup.trace('id_act', id_hoi=id_hoi) for id_hoi in ids_hoi]
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
      return sorted(self.lookup.retrieve('ids_sact'))

    ids_sact_intersection = []

    # split
    if split is not None:
      assert split in self.lookup.retrieve('splits')
      ids_sact = self.get_ids_sact(ids_act=self.lookup.retrieve('ids_act', split))
      ids_sact_intersection.append(ids_sact)

    # cnames_sact
    if cnames_sact is not None:
      ids_sact = []
      for id_sact in self.lookup.retrieve('ids_sact'):
        ann_sact = self.lookup.retrieve('ann_sact', id_sact)
        if ann_sact.cname in cnames_sact:
          ids_sact.append(id_sact)
      ids_sact_intersection.append(ids_sact)

    # ids_act
    if ids_act is not None:
      ids_sact = itertools.chain(*[self.lookup.trace('ids_sact', id_act=id_act) for id_act in ids_act])
      ids_sact_intersection.append(ids_sact)

    # ids_hoi
    if ids_hoi is not None:
      ids_sact = [self.lookup.trace('id_sact', id_hoi=id_hoi) for id_hoi in ids_hoi]
      ids_sact_intersection.append(ids_sact)

    # cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel
    if not all(x is None for x in [cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel]):
      kwargs = {'cnames_actor': cnames_actor, 'cnames_object': cnames_object,
                'cnames_ia': cnames_ia, 'cnames_ta': cnames_ta,
                'cnames_att': cnames_att, 'cnames_rel': cnames_rel}
      ids_sact = [self.lookup.trace('id_sact', id_hoi=id_hoi) for id_hoi in self.get_ids_hoi(**kwargs)]
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
      return sorted(self.lookup.retrieve('ids_hoi'))

    ids_hoi_intersection = []

    # split
    if split is not None:
      assert split in self.lookup.retrieve('splits')
      ids_hoi = self.get_ids_hoi(ids_act=self.lookup.retrieve('ids_act', split))
      ids_hoi_intersection.append(ids_hoi)

    # ids_act
    if ids_act is not None:
      ids_hoi = itertools.chain(*[self.lookup.trace('ids_hoi', id_act=id_act) for id_act in ids_act])
      ids_hoi_intersection.append(ids_hoi)

    # ids_sact
    if ids_sact is not None:
      ids_hoi = itertools.chain(*[self.lookup.trace('ids_hoi', id_sact=id_sact) for id_sact in ids_sact])
      ids_hoi_intersection.append(ids_hoi)

    # cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel
    cnames_dict = {'actors': cnames_actor, 'objects': cnames_object,
                   'ias': cnames_ia, 'tas': cnames_ta,
                   'atts': cnames_att, 'rels': cnames_rel}
    for var, cnames in cnames_dict.items():
      if cnames is not None:
        ids_hoi = []
        for id_hoi in self.lookup.retrieve('ids_hoi'):
          ann_hoi = self.lookup.retrieve('ann_hoi', id_hoi)
          if not set(cnames).isdisjoint([x.cname for x in getattr(ann_hoi, var)]):
            ids_hoi.append(id_hoi)
        ids_hoi_intersection.append(ids_hoi)

    ids_hoi_intersection = sorted(set.intersection(*map(set, ids_hoi_intersection)))
    return ids_hoi_intersection

  def get_metadata(self, ids_act: list) -> list:
    return [self.lookup.retrieve('metadatum', id_act) for id_act in ids_act]

  def get_anns_act(self, ids_act: list) -> list:
    return [self.lookup.retrieve('ann_act', id_act) for id_act in ids_act]
  
  def get_anns_sact(self, ids_sact: list) -> list:
     return [self.lookup.retrieve('ann_sact', id_sact) for id_sact in ids_sact]
    
  def get_anns_hoi(self, ids_hoi: list) -> list:
     return [self.lookup.retrieve('ann_hoi', id_hoi) for id_hoi in ids_hoi]

  def get_paths(self,
                ids_act: list=None,
                ids_sact: list=None,
                ids_hoi: list=None,
                sanity_check: bool=True) -> list:
    assert sum([x is not None for x in [ids_act, ids_sact, ids_hoi]]) == 1

    if ids_act is not None:
      paths = [osp.join(self.dir_moma, f"videos/activity{'_fr' if self.full_res else ''}/{id_act}.mp4")
               for id_act in ids_act]
    elif ids_sact is not None:
      paths = [osp.join(self.dir_moma, f"videos/sub_activity{'_fr' if self.full_res else ''}/{id_sact}.mp4")
               for id_sact in ids_sact]
    elif ids_hoi is not None:
      paths = [osp.join(self.dir_moma, f'videos/interaction/{id_hoi}.jpg') for id_hoi in ids_hoi]
    else:
      assert id_hoi_clip is not None
      clip = self.get_clips(ids_hoi=[id_hoi_clip])[0]
      times = [x[1] for x in clip.neighbors]+[clip.time]
      paths = [osp.join(self.dir_moma, f'videos/interaction_frames/{x[0]}.jpg') for x in clip.neighbors]+ \
              [osp.join(self.dir_moma, f'videos/interaction/{id_hoi_clip}.jpg')]
      paths = [x for _, x in sorted(zip(times, paths))]

    if sanity_check and not all(osp.exists(path) for path in paths):
      paths_missing = [path for path in paths if not osp.exists(path)]
      paths_missing = paths_missing[:5] if len(paths_missing) > 5 else paths_missing
      assert False, f'{len(paths_missing)} paths do not exist: {paths_missing}'

    return paths

  def get_paths_window(self, id_hoi):
    """ Given a higher-order interaction ID, return
     - a path to the 1s video clip centered at the higher-order interaction (<1s if exceeds the raw video boundary)
     - paths to 5 frames centered at the higher-order interaction (<5 frames if exceeds the raw video boundary)
    """
    window = self.lookup.retrieve('window', id_hoi)
    now = self.get_anns_hoi(ids_hoi=[id_hoi])[0].time
    window = [[os.path.join(self.dir_moma, f'videos/interaction_frames/{fname}.jpg'), time] for fname, time in window]+\
             [[os.path.join(self.dir_moma, f'videos/interaction/{id_hoi}.jpg'), now]]
    window = sorted(window, key=lambda x: x[1])

    path_video = os.path.join(self.dir_moma, f'videos/interaction_video/{id_hoi}.mp4')
    paths_frame = [path_frame for path_frame, time in window]

    return path_video, paths_frame

  def sort(self, ids_sact: list=None, ids_hoi: list=None, sanity_check: bool=True):
    """ Given a list of sub-activity or higher-order interaction instance IDs, return them in sorted order
    """
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
