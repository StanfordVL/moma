import itertools
import json
import numpy as np
import os
import random

from .data import *

"""
The following functions are defined:
 - get_taxonomy: Get the taxonomy of a concept ('act', 'sact', etc.)
 - get_cnames: Get the class name of a concept ('act', 'sact', etc.) that satisfy certain conditions
 - get_ids_act: Get the unique activity instance IDs that satisfy certain conditions
 - get_ids_sact: Get the unique sub-activity instance IDs that satisfy certain conditions
 - get_ids_hoi: Get the unique higher-order interaction instance IDs that satisfy certain conditions
 - get_ids_actor: Get the unique actor global instance IDs that satisfy certain conditions
 - get_ids_object: Get the unique object global instance IDs that satisfy certain conditions
 - get_metadata: Given activity instance IDs, return the metadata of the associated raw videos
 - get_anns_act: Given activity instance IDs, return their annotations
 - get_anns_sact: Given sub-activity instance IDs, return their annotations
 - get_anns_hoi: Given higher-order interaction instance IDs, return their annotations
 - get_anns_actor: Given actor global instance IDs, return their annotations
 - get_anns_object: Given object global instance IDs, return their annotations
 - get_paths: Given instance IDs, return data paths
 
Acronyms:
 - act: activity
 - sact: sub-activity
 - hoi: higher-order interaction
 - ent: entity
 - ia: intransitive action
 - ta: transitive action
 - att: attribute
 - rel: relationship
 - ann: annotation
 - id: instance ID
 - cname: class name
 - cid: class ID
"""


class MOMA:
  def __init__(self, dir_moma: str, toy: bool=False, generate_split: bool=False):
    """
     - toy: load a toy annotation file to quickly illustrate the behavior of the various algorithms
     - generate_split: generate a new train/val split
    """
    assert os.path.isdir(os.path.join(dir_moma, 'anns')) and os.path.isdir(os.path.join(dir_moma, 'videos'))

    self.dir_moma = dir_moma
    self.toy = toy
    self.taxonomy = self.__read_taxonomy()
    
    self.metadata, \
    self.id_act_to_ann_act, self.id_sact_to_ann_sact, self.id_hoi_to_ann_hoi, \
    self.id_actor_to_ann_actor, self.id_object_to_ann_object, \
    self.id_sact_to_id_act, self.id_hoi_to_id_sact, \
    self.id_actor_to_id_hoi, self.id_object_to_id_hoi = self.__read_anns()

    self.ids_act_train, self.ids_act_val = self.__read_splits(generate_split)
    self.statistics, self.distributions = self.__get_summary()
    self.statistics_train, self.distributions_train = self.__get_summary('train')
    self.statistics_val, self.distributions_val = self.__get_summary('val')

  def get_taxonomy(self, concept):
    assert concept in self.taxonomy
    return self.taxonomy[concept]

  def get_cnames(self, concept, num_instances=None, split=None):
    assert concept in ['actor', 'object']

    if num_instances is None:
      return self.get_taxonomy(concept)

    if split == 'train':
      distribution = self.distributions_train[concept]
    elif split == 'val':
      distribution = self.distributions_val[concept]
    else:
      assert split is None
      distribution = self.distributions[concept]

    cnames = []
    for i, cname in enumerate(self.get_taxonomy(concept)):
      if distribution[i] >= num_instances:
        cnames.append(cname)
    return cnames

  def get_ids_act(self, split: str=None, cnames_act: list[str]=None,
                  ids_sact: list[str]=None, ids_hoi: list[str]=None) -> list[str]:
    """ Get the unique activity instance IDs that satisfy certain conditions
    dataset split
     - split: get activity IDs [ids_act] that belong to the given dataset split [split='train' or 'val']
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
      if split == 'train':
        ids_act_intersection.append(self.ids_act_train)
      else:
        assert split == 'val'
        ids_act_intersection.append(self.ids_act_val)

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
      ids_act = itertools.chain(*[self.id_sact_to_id_act[self.id_hoi_to_id_sact[id_hoi]] for id_hoi in ids_hoi])
      ids_act_intersection.append(ids_act)

    ids_act_intersection = sorted(set.intersection(*map(set, ids_act_intersection)))
    return ids_act_intersection

  def get_ids_sact(self, split: str=None,
                   cnames_sact: list[str]=None, ids_act: list[str]=None, ids_hoi: list[str]=None,
                   cnames_actor: list[str]=None, cnames_object: list[str]=None,
                   cnames_ia: list[str]=None, cnames_ta: list[str]=None,
                   cnames_att: list[str]=None, cnames_rel: list[str]=None) -> list[str]:
    """ Get the unique sub-activity instance IDs that satisfy certain conditions
    dataset split
     - split: get sub-activity IDs [ids_sact] that belong to the given dataset split [split='train' or 'val']
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
      if split == 'train':
        ids_sact = self.get_ids_sact(ids_act=self.ids_act_train)
      else:
        assert split == 'val'
        ids_sact = self.get_ids_sact(ids_act=self.ids_act_val)
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
                  ids_act: list[str]=None, ids_sact: list[str]=None,
                  cnames_actor: list[str]=None, cnames_object: list[str]=None,
                  cnames_ia: list[str]=None, cnames_ta: list[str]=None,
                  cnames_att: list[str]=None, cnames_rel: list[str]=None) -> list[str]:
    """ Get the unique higher-order interaction instance IDs that satisfy certain conditions
    dataset split
     - split: get higher-order interaction IDs [ids_hoi] that belong to the given dataset split [split='train' or 'val']
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
      if split == 'train':
        ids_hoi = self.get_ids_hoi(ids_act=self.ids_act_train)
      else:
        assert split == 'val'
        ids_hoi = self.get_ids_hoi(ids_act=self.ids_act_val)
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

  def get_ids_actor(self, split: str=None, cnames_actor: list[str]=None,
                    ids_act: list[str]=None, ids_sact: list[str]=None, ids_hoi: list[str]=None) -> list[str]:
    """ Get the unique actor global instance IDs that satisfy certain conditions
    dataset split
     - split: get actor global IDs [ids_hoi] that belong to the given dataset split [split='train' or 'val']
    same-level
     - cnames_actor: get sub-activity IDs [ids_sact] for given sub-activity class names [cnames_sact]
    top-down
     - ids_act: get actor global IDs [ids_hoi] for given activity IDs [ids_act]
     - ids_sact: get actor global IDs [ids_hoi] for given sub-activity IDs [ids_sact]
     - ids_hoi: get actor global IDs [ids_hoi] for given higher-order interaction IDs [ids_hoi]
    """
    if all(x is None for x in [split, ids_act, ids_sact, ids_hoi, cnames_actor]):
      return sorted(self.id_actor_to_ann_actor.keys())

    ids_actor_intersection = []
    
    # split
    if split is not None:
      if split == 'train':
        ids_actor = self.get_ids_actor(ids_act=self.ids_act_train)
      else:
        assert split == 'val'
        ids_actor = self.get_ids_actor(ids_act=self.ids_act_val)
      ids_actor_intersection.append(ids_actor)

    # cnames_act
    if cnames_actor is not None:
      ids_actor = []
      for id_actor, ann_actor in self.id_actor_to_ann_actor.items():
        if ann_actor.cname in cnames_actor:
          ids_actor.append(id_actor)
      ids_actor_intersection.append(ids_actor)

    # ids_act
    if ids_act is not None:
      ids_actor = itertools.chain(*[self.id_actor_to_id_hoi.inverse[id_hoi]
                                    if id_hoi in self.id_actor_to_id_hoi.inverse else []
                                    for id_act in ids_act
                                    for id_sact in self.id_sact_to_id_act.inverse[id_act]
                                    for id_hoi in self.id_hoi_to_id_sact.inverse[id_sact]])
      ids_actor_intersection.append(ids_actor)

    # ids_sact
    if ids_sact is not None:
      ids_actor = itertools.chain(*[self.id_actor_to_id_hoi.inverse[id_hoi]
                                    if id_hoi in self.id_actor_to_id_hoi.inverse else []
                                    for id_sact in ids_sact
                                    for id_hoi in self.id_hoi_to_id_sact.inverse[id_sact]])
      ids_actor_intersection.append(ids_actor)

    # ids_hoi
    if ids_hoi is not None:
      ids_actor = [self.id_actor_to_id_hoi.inverse[id_hoi] for id_hoi in ids_hoi]
      ids_actor_intersection.append(ids_actor)
      
    ids_actor_intersection = sorted(set.intersection(*map(set, ids_actor_intersection)))
    return ids_actor_intersection

  def get_ids_object(self, split: str=None, cnames_object: list[str]=None,
                     ids_act: list[str]=None, ids_sact: list[str]=None, ids_hoi: list[str]=None) -> list[str]:
    """ Get the unique object global instance IDs that satisfy certain conditions
    dataset split
     - split: get object global IDs [ids_hoi] that belong to the given dataset split [split='train' or 'val']
    same-level
     - cnames_object: get sub-activity IDs [ids_sact] for given sub-activity class names [cnames_sact]
    top-down
     - ids_act: get object global IDs [ids_hoi] for given activity IDs [ids_act]
     - ids_sact: get object global IDs [ids_hoi] for given sub-activity IDs [ids_sact]
     - ids_hoi: get object global IDs [ids_hoi] for given higher-order interaction IDs [ids_hoi]
    """
    if all(x is None for x in [split, ids_act, ids_sact, ids_hoi, cnames_object]):
      return sorted(self.id_object_to_ann_object.keys())

    ids_object_intersection = []

    # split
    if split is not None:
      if split == 'train':
        ids_object = self.get_ids_object(ids_act=self.ids_act_train)
      else:
        assert split == 'val'
        ids_object = self.get_ids_object(ids_act=self.ids_act_val)
      ids_object_intersection.append(ids_object)

    # cnames_act
    if cnames_object is not None:
      ids_object = []
      for id_object, ann_object in self.id_object_to_ann_object.items():
        if ann_object.cname in cnames_object:
          ids_object.append(id_object)
      ids_object_intersection.append(ids_object)

    # ids_act
    if ids_act is not None:
      ids_object = itertools.chain(*[self.id_object_to_id_hoi.inverse[id_hoi]
                                     if id_hoi in self.id_object_to_id_hoi.inverse else []
                                     for id_act in ids_act
                                     for id_sact in self.id_sact_to_id_act.inverse[id_act]
                                     for id_hoi in self.id_hoi_to_id_sact.inverse[id_sact]])
      ids_object_intersection.append(ids_object)

    # ids_sact
    if ids_sact is not None:
      ids_object = itertools.chain(*[self.id_object_to_id_hoi.inverse[id_hoi]
                                     if id_hoi in self.id_object_to_id_hoi.inverse else []
                                     for id_sact in ids_sact
                                     for id_hoi in self.id_hoi_to_id_sact.inverse[id_sact]])
      ids_object_intersection.append(ids_object)

    # ids_hoi
    if ids_hoi is not None:
      ids_object = [self.id_object_to_id_hoi.inverse[id_hoi] for id_hoi in ids_hoi]
      ids_object_intersection.append(ids_object)
      
    ids_object_intersection = sorted(set.intersection(*map(set, ids_object_intersection)))
    return ids_object_intersection

  def get_metadata(self, ids_act: list[str]) -> list[Metadatum]:
    return [self.metadata[id_act] for id_act in ids_act]

  def get_anns_act(self, ids_act: list[str]) -> list[Act]:
    return [self.id_act_to_ann_act[id_act] for id_act in ids_act]

  def get_anns_sact(self, ids_sact: list[str]) -> list[SAct]:
    return [self.id_sact_to_ann_sact[id_sact] for id_sact in ids_sact]

  def get_anns_hoi(self, ids_hoi: list[str]) -> list[HOI]:
    return [self.id_hoi_to_ann_hoi[id_hoi] for id_hoi in ids_hoi]

  def get_anns_actor(self, ids_actor: list[str]) -> list[Ent]:
    return [self.id_actor_to_ann_actor[id_actor] for id_actor in ids_actor]

  def get_anns_object(self, ids_object: list[str]) -> list[Ent]:
    return [self.id_object_to_ann_object[id_object] for id_object in ids_object]

  def get_paths(self, ids_act: list[str]=None, ids_sact: list[str]=None, ids_hoi: list[str]=None) -> list[str]:
    assert sum([x is not None for x in [ids_act, ids_sact, ids_hoi]]) == 1

    if ids_act is not None:
      paths = [os.path.join(self.dir_moma, f'videos/activity/{id_act}.mp4') for id_act in ids_act]
    elif ids_sact is not None:
      paths = [os.path.join(self.dir_moma, f'videos/sub_activity/{id_sact}.mp4') for id_sact in ids_sact]
    else:
      paths = [os.path.join(self.dir_moma, f'videos/higher_order_interaction/{id_hoi}.jpg') for id_hoi in ids_hoi]

    assert all(os.path.exists(path) for path in paths)
    return paths

  def __read_taxonomy(self):
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/actor.json'), 'r') as f:
      taxonomy_actor = json.load(f)
      taxonomy_actor = sorted(itertools.chain(*taxonomy_actor.values()))
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/object.json'), 'r') as f:
      taxonomy_object = json.load(f)
      taxonomy_object = sorted(itertools.chain(*taxonomy_object.values()))
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/intransitive_action.json'), 'r') as f:
      taxonomy_ia = json.load(f)
      taxonomy_ia = sorted(map(tuple, itertools.chain(*taxonomy_ia.values())))
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/transitive_action.json'), 'r') as f:
      taxonomy_ta = json.load(f)
      taxonomy_ta = sorted(map(tuple, itertools.chain(*taxonomy_ta.values())))
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/attribute.json'), 'r') as f:
      taxonomy_att = json.load(f)
      taxonomy_att = sorted(map(tuple, itertools.chain(*taxonomy_att.values())))
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/relationship.json'), 'r') as f:
      taxonomy_rel = json.load(f)
      taxonomy_rel = sorted(map(tuple, itertools.chain(*taxonomy_rel.values())))
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/act_sact.json'), 'r') as f:
      taxonomy_act_sact = json.load(f)
      taxonomy_act = sorted(taxonomy_act_sact.keys())
      taxonomy_sact = sorted(itertools.chain(*taxonomy_act_sact.values()))
      taxonomy_sact_to_act = bidict({cname_sact: cname_act for cname_act, cnames_sact in taxonomy_act_sact.items()
                                                           for cname_sact in cnames_sact})

    taxonomy = {
      'actor': taxonomy_actor,
      'object': taxonomy_object,
      'ia': taxonomy_ia,
      'ta': taxonomy_ta,
      'att': taxonomy_att,
      'rel': taxonomy_rel,
      'act': taxonomy_act,
      'sact': taxonomy_sact,
      'sact_to_act': taxonomy_sact_to_act
    }

    return taxonomy

  def __read_anns(self):
    fname = 'anns_toy.json' if self.toy else 'anns.json'
    with open(os.path.join(self.dir_moma, f'anns/{fname}'), 'r') as f:
      anns_raw = json.load(f)

    metadata, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi, \
        id_actor_to_ann_actor, id_object_to_ann_object = {}, {}, {}, {}, {}, {}
    id_sact_to_id_act, id_hoi_to_id_sact, id_actor_to_id_hoi, id_object_to_id_hoi = {}, {}, {}, {}

    for ann_raw in anns_raw:
      ann_act_raw = ann_raw['activity']
      metadata[ann_act_raw['id']] = Metadatum(ann_raw)
      id_act_to_ann_act[ann_act_raw['id']] = Act(ann_act_raw, self.taxonomy['act'])
      anns_sact_raw = ann_act_raw['sub_activities']

      for ann_sact_raw in anns_sact_raw:
        id_sact_to_ann_sact[ann_sact_raw['id']] = SAct(ann_sact_raw, self.taxonomy['sact'])
        id_sact_to_id_act[ann_sact_raw['id']] = ann_act_raw['id']
        anns_hoi_raw = ann_sact_raw['higher_order_interactions']

        for ann_hoi_raw in anns_hoi_raw:
          id_hoi_to_ann_hoi[ann_hoi_raw['id']] = HOI(ann_hoi_raw,
                                                     self.taxonomy['actor'], self.taxonomy['object'],
                                                     self.taxonomy['ia'], self.taxonomy['ta'],
                                                     self.taxonomy['att'], self.taxonomy['rel'])
          id_hoi_to_id_sact[ann_hoi_raw['id']] = ann_sact_raw['id']
          anns_actor_raw = ann_hoi_raw['actors']
          anns_object_raw = ann_hoi_raw['objects']

          for ann_actor_raw in anns_actor_raw:
            id_actor = f"{ann_hoi_raw['id']}_{ann_actor_raw['id']}"
            id_actor_to_ann_actor[id_actor] = Ent(ann_actor_raw, 'actor', self.taxonomy['actor'])
            id_actor_to_id_hoi[id_actor] = ann_hoi_raw['id']
            
          for ann_object_raw in anns_object_raw:
            id_object = f"{ann_hoi_raw['id']}_{ann_object_raw['id']}"
            id_object_to_ann_object[id_object] = Ent(ann_object_raw, 'object', self.taxonomy['object'])
            id_object_to_id_hoi[id_object] = ann_hoi_raw['id']

    id_sact_to_id_act = bidict(id_sact_to_id_act)
    id_hoi_to_id_sact = bidict(id_hoi_to_id_sact)
    id_actor_to_id_hoi = bidict(id_actor_to_id_hoi)
    id_object_to_id_hoi = bidict(id_object_to_id_hoi)

    return metadata, \
           id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi, id_actor_to_ann_actor, id_object_to_ann_object, \
           id_sact_to_id_act, id_hoi_to_id_sact, id_actor_to_id_hoi, id_object_to_id_hoi

  def __read_splits(self, generate_split=False):
    if generate_split:
      self.__generate_splits()

    path_split = os.path.join(self.dir_moma, 'anns/split.json')

    if not os.path.isfile(path_split):
      print(f'Dataset split file does not exist: {path_split}')
      return None, None

    with open(path_split, 'r') as f:
      ids_act_splits = json.load(f)

    ids_act_train, ids_act_val = ids_act_splits['train'], ids_act_splits['val']

    if self.toy:
      ids_act_train = list(set(self.get_ids_act())-set(ids_act_val))
      ids_act_val = list(set(self.get_ids_act())-set(ids_act_train))
    else:
      assert set(self.get_ids_act()) == set(ids_act_train+ids_act_val)

    return ids_act_train, ids_act_val

  def __generate_splits(self, ratio_train=0.80):
    ids_act = sorted(self.id_act_to_ann_act.keys())
    ids_act = random.sample(ids_act, len(ids_act))

    size_train = round(len(ids_act)*ratio_train)
    ids_act_train = ids_act[:size_train]
    ids_act_val = ids_act[size_train:]

    path_split = os.path.join(self.dir_moma, 'anns/split.json')
    with open(path_split, 'w') as f:
      json.dump({'train': ids_act_train, 'val': ids_act_val}, f, indent=4, sort_keys=True)

  def __get_summary(self, split=None):
    if split is None:
      metadata = self.metadata.values()
      anns_act = self.id_act_to_ann_act.values()
      anns_sact = self.id_sact_to_ann_sact.values()
      anns_hoi = self.id_hoi_to_ann_hoi.values()
    elif split == 'train' or split == 'val':
      ids_act = self.ids_act_train if split == 'train' else self.ids_act_val
      metadata = self.get_metadata(ids_act=ids_act)
      anns_act = self.get_anns_act(ids_act=ids_act)
      ids_sact = self.get_ids_sact(ids_act=ids_act)
      anns_sact = self.get_anns_sact(ids_sact=ids_sact)
      ids_hoi = self.get_ids_hoi(ids_act=ids_act)
      anns_hoi = self.get_anns_hoi(ids_hoi=ids_hoi)
    else:
      assert False
  
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
