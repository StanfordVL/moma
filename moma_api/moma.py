import itertools
import json
import numpy as np
import os

from .data import *


class MOMA:
  def __init__(self, dir_moma: str):
    self.dir_moma = dir_moma
    self.taxonomy = self.__read_taxonomy()
    self.metadata, self.id_act_to_ann_act, self.id_sact_to_ann_sact, self.id_hoi_to_ann_hoi, \
        self.id_sact_to_id_act, self.id_hoi_to_id_sact = self.__read_anns()
    self.ids_act_train, self.ids_act_val = self.__read_splits()

  def get_stats(self, split=None):
    if split is None:
      anns_act = self.id_act_to_ann_act.values()
      anns_sact = self.id_sact_to_ann_sact.values()
      anns_hoi = self.id_hoi_to_ann_hoi.values()
    elif split == 'train' or split == 'val':
      ids_act = self.ids_act_train if split == 'train' else self.ids_act_val
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

    stats_overall = {
      'activity': {
        'num_instances': num_acts,
        'num_classes': num_classes_act
      },
      'sub_activity': {
        'num_instances': num_sacts,
        'num_classes': num_classes_sact
      },
      'higher_order_interaction': {
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
      'intransitive_action': {
        'num_instances': num_ias,
        'num_classes': num_classes_ia
      },
      'transitive_action': {
        'num_instances': num_tas,
        'num_classes': num_classes_ta
      },
      'attribute': {
        'num_instances': num_atts,
        'num_classes': num_classes_att
      },
      'relationship': {
        'num_instances': num_rels,
        'num_classes': num_classes_rel
      },
    }

    stats_per_class = {
      'activity': {
        'counts': bincount_act,
        'class_names': self.taxonomy['act']
      },
      'sub_activity': {
        'counts': bincount_sact,
        'class_names': self.taxonomy['sact']
      },
      'actor': {
        'counts': bincount_actor,
        'class_names': self.taxonomy['actor']
      },
      'object': {
        'counts': bincount_object,
        'class_names': self.taxonomy['object']
      },
      'intransitive_action': {
        'counts': bincount_ia,
        'class_names': [x[0] for x in self.taxonomy['ia']]
      },
      'transitive_action': {
        'counts': bincount_ta,
        'class_names': [x[0] for x in self.taxonomy['ta']]
      },
      'attribute': {
        'counts': bincount_att,
        'class_names': [x[0] for x in self.taxonomy['att']]
      },
      'relationship': {
        'counts': bincount_rel,
        'class_names': [x[0] for x in self.taxonomy['rel']]
      }
    }

    return stats_overall, stats_per_class

  def get_ids_act(self, split: str=None, cnames_act: list[str]=None,
                  ids_sact: list[str]=None, ids_hoi: list[str]=None) -> list[str]:
    """ Get the unique IDs of activity instances that satisfy certain conditions
    dataset split
     - split: get activity IDs [ids_act] that belong to the given dataset split [split='train' or 'val]
    same-level
     - cnames_act: get activity IDs [ids_act] for given activity class names [cnames_act]
    bottom-up
     - ids_sact: get activity IDs [ids_act] for given sub-activity IDs [ids_sact]
     - ids_hoi: get activity IDs [ids_act] for given higher-order interaction IDs [ids_hoi]
    """
    if all(x is None for x in [cnames_act, ids_sact, ids_hoi]):
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
    """ Get the unique IDs of sub-activity instances that satisfy certain conditions
    dataset split
     - split: get sub-activity IDs [ids_sact] that belong to the given dataset split [split='train' or 'val]
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
    if all(x is None for x in [cnames_sact, ids_act, ids_hoi, cnames_actor, cnames_object,
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
    """ Get the unique IDs of higher-order interaction instances that satisfy certain conditions
    dataset split
     - split: get higher-order interaction IDs [ids_hoi] that belong to the given dataset split [split='train' or 'val]
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
    if all(x is None for x in [ids_act, ids_sact, cnames_actor, cnames_object,
                               cnames_ia, cnames_ta, cnames_att, cnames_rel]):
      return sorted(self.id_hoi_to_ann_hoi.keys())

    ids_hoi_interaction = []

    # split
    if split is not None:
      if split == 'train':
        ids_hoi = self.get_ids_hoi(ids_act=self.ids_act_train)
      else:
        assert split == 'val'
        ids_hoi = self.get_ids_hoi(ids_act=self.ids_act_val)
      ids_hoi_interaction.append(ids_hoi)

    # ids_act
    if ids_act is not None:
      ids_hoi = itertools.chain(*[self.id_hoi_to_id_sact.inverse[id_sact]
                                  for id_act in ids_act
                                  for id_sact in self.id_sact_to_id_act.inverse[id_act]])
      ids_hoi_interaction.append(ids_hoi)

    # ids_sact
    if ids_sact is not None:
      ids_hoi = itertools.chain(*[self.id_hoi_to_id_sact.inverse[id_sact] for id_sact in ids_sact])
      ids_hoi_interaction.append(ids_hoi)

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
        ids_hoi_interaction.append(ids_hoi)

    ids_hoi_interaction = sorted(set.intersection(*map(set, ids_hoi_interaction)))
    return ids_hoi_interaction

  def get_ann_act(self, id_act: str) -> Act:
    return self.id_act_to_ann_act[id_act]

  def get_ann_sact(self, id_sact: str) -> SAct:
    return self.id_sact_to_ann_sact[id_sact]

  def get_ann_hoi(self, id_hoi: str) -> HOI:
    return self.id_hoi_to_ann_hoi[id_hoi]

  def get_anns_act(self, ids_act: list[str]) -> list[Act]:
    return [self.id_act_to_ann_act[id_act] for id_act in ids_act]

  def get_anns_sact(self, ids_sact: list[str]) -> list[SAct]:
    return [self.id_sact_to_ann_sact[id_sact] for id_sact in ids_sact]

  def get_anns_hoi(self, ids_hoi: list[str]) -> list[HOI]:
    return [self.id_hoi_to_ann_hoi[id_hoi] for id_hoi in ids_hoi]

  def get_path(self, id_act: str=None, id_sact: str=None, id_hoi: str=None) -> str:
    assert sum([x is not None for x in [id_act, id_sact, id_hoi]]) == 1

    if id_act is not None:
      path = os.path.join(self.dir_moma, f'videos/activity/{id_act}.mp4')
    elif id_sact is not None:
      path = os.path.join(self.dir_moma, f'videos/sub_activity/{id_sact}.mp4')
    else:
      path = os.path.join(self.dir_moma, f'videos/higher_order_interaction/{id_hoi}.jpg')

    assert os.path.exists(path), path
    return path

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
    with open(os.path.join(self.dir_moma, 'anns/anns.json'), 'r') as f:
      anns_raw = json.load(f)

    metadata, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi = {}, {}, {}, {}
    id_sact_to_id_act, id_hoi_to_id_sact = {}, {}

    for ann_raw in anns_raw:
      ann_act_raw = ann_raw['activity']
      metadata[ann_act_raw['id']] = Metadata(ann_raw)
      id_act_to_ann_act[ann_act_raw['id']] = Act(ann_act_raw, self.taxonomy['act'])
      anns_sact_raw = ann_act_raw['sub_activities']

      for ann_sact_raw in anns_sact_raw:
        id_sact_to_ann_sact[ann_sact_raw['id']] = SAct(ann_sact_raw, self.taxonomy['sact'])
        anns_hoi_raw = ann_sact_raw['higher_order_interactions']
        id_sact_to_id_act[ann_sact_raw['id']] = ann_act_raw['id']

        for ann_hoi_raw in anns_hoi_raw:
          id_hoi_to_ann_hoi[ann_hoi_raw['id']] = HOI(ann_hoi_raw,
                                            self.taxonomy['actor'], self.taxonomy['object'],
                                            self.taxonomy['ia'], self.taxonomy['ta'],
                                            self.taxonomy['att'], self.taxonomy['rel'])
          id_hoi_to_id_sact[ann_hoi_raw['id']] = ann_sact_raw['id']

    id_sact_to_id_act = bidict(id_sact_to_id_act)
    id_hoi_to_id_sact = bidict(id_hoi_to_id_sact)

    return metadata, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi, id_sact_to_id_act, id_hoi_to_id_sact

  def __read_splits(self):
    path_split = os.path.join(self.dir_moma, 'anns/split.json')

    # dataset split file deos not exist, please run preproc.py
    if not os.path.isfile(path_split):
      return None, None

    with open(path_split, 'r') as f:
      ids_act_splits = json.load(f)

    ids_act_train, ids_act_val = ids_act_splits['train'], ids_act_splits['val']
    return ids_act_train, ids_act_val
