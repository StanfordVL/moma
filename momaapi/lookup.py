import itertools
import json
import os
import pickle

from .data import bidict, lazydict, Metadatum, Act, SAct, HOI


class Lookup:
  def __init__(self, dir_moma, taxonomy):
    self.dir_moma = dir_moma
    self.taxonomy = taxonomy

    self.id_act_to_metadatum = None
    self.id_act_to_ann_act = None
    self.id_sact_to_ann_sact = None
    self.id_hoi_to_ann_hoi = None
    self.id_sact_to_id_act = None
    self.id_hoi_to_id_sact = None
    self.id_hoi_to_window = None

    self.read_anns()

  @staticmethod
  def save_cache(dir_moma,
                 id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi,
                 id_sact_to_id_act, id_hoi_to_id_sact, id_hoi_to_window):
    named_variables = {
      'id_act_to_metadatum': id_act_to_metadatum,
      'id_act_to_ann_act': id_act_to_ann_act,
      'id_sact_to_ann_sact': id_sact_to_ann_sact,
      'id_hoi_to_ann_hoi': id_hoi_to_ann_hoi,
      'id_sact_to_id_act': id_sact_to_id_act,
      'id_hoi_to_id_sact': id_hoi_to_id_sact,
      'id_hoi_to_window': id_hoi_to_window
    }

    os.makedirs(os.path.join(dir_moma, 'anns/cache'), exist_ok=True)
    os.makedirs(os.path.join(dir_moma, 'anns/cache/id_hoi_to_ann_hoi'), exist_ok=True)

    for name, variable in named_variables.items():
      assert variable is not None

      if name == 'id_hoi_to_ann_hoi':
        for id_hoi, ann_hoi in variable.items():
          with open(os.path.join(dir_moma, 'anns/cache/id_hoi_to_ann_hoi', id_hoi), 'wb') as f:
            pickle.dump(ann_hoi, f)

      else:
        with open(os.path.join(dir_moma, 'anns/cache', name), 'wb') as f:
          pickle.dump(variable, f)

  @staticmethod
  def load_cache(dir_moma):
    variables = []
    for name in ['id_act_to_metadatum', 'id_act_to_ann_act', 'id_sact_to_ann_sact',
                 'id_sact_to_id_act', 'id_hoi_to_id_sact', 'id_hoi_to_window']:
      with open(os.path.join(dir_moma, 'anns/cache', name), 'rb') as f:
        variable = pickle.load(f)
      variables.append(variable)

    ids_hoi = variables[4].keys()
    id_hoi_to_ann_hoi = lazydict(ids_hoi, os.path.join(dir_moma, 'anns/cache/id_hoi_to_ann_hoi'))
    variables.insert(3, id_hoi_to_ann_hoi)

    return variables

  def read_anns(self):
    try:
      id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi, \
      id_sact_to_id_act, id_hoi_to_id_sact, id_hoi_to_window = self.load_cache(self.dir_moma)

    except FileNotFoundError:
      print('FileNotFoundError')
      with open(os.path.join(self.dir_moma, f'anns/anns.json'), 'r') as f:
        anns_raw = json.load(f)

      id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi = {}, {}, {}, {}
      id_sact_to_id_act, id_hoi_to_id_sact = {}, {}

      for ann_raw in anns_raw:
        ann_act_raw = ann_raw['activity']
        id_act_to_metadatum[ann_act_raw['id']] = Metadatum(ann_raw)
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

      with open(os.path.join(self.dir_moma, f'videos/interaction_frames/timestamps.json'), 'r') as f:
        id_hoi_to_window = json.load(f)

      self.save_cache(self.dir_moma,
                      id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi,
                      id_sact_to_id_act, id_hoi_to_id_sact, id_hoi_to_window)

    id_sact_to_id_act = bidict(id_sact_to_id_act)
    id_hoi_to_id_sact = bidict(id_hoi_to_id_sact)

    self.id_act_to_metadatum = id_act_to_metadatum
    self.id_act_to_ann_act = id_act_to_ann_act
    self.id_sact_to_ann_sact = id_sact_to_ann_sact
    self.id_hoi_to_ann_hoi = id_hoi_to_ann_hoi
    self.id_sact_to_id_act = id_sact_to_id_act
    self.id_hoi_to_id_sact = id_hoi_to_id_sact
    self.id_hoi_to_window = id_hoi_to_window

  def get_ids(self, level):
    assert level in ['act', 'sact', 'hoi']

    if level == 'act':
      return self.id_act_to_ann_act.keys()
    elif level == 'sact':
      return self.id_sact_to_ann_sact.keys()
    elif level == 'hoi':
      return self.id_hoi_to_ann_hoi.keys()
    else:
      raise ValueError

  def get_ann(self, id_act=None, id_sact=None, id_hoi=None):
    assert sum([x is not None for x in [id_act, id_sact, id_hoi]]) == 1

    if id_act is not None:
      return self.id_act_to_ann_act[id_act]
    elif id_sact is not None:
      return self.id_sact_to_ann_sact[id_sact]
    elif id_hoi is not None:
      return self.id_hoi_to_ann_hoi[id_hoi]
    else:
      raise ValueError

  def get_metadatum(self, id_act):
    return self.id_act_to_metadatum[id_act]

  def get_window(self, id_hoi):
    return self.id_hoi_to_window[id_hoi]

  def trace(self, id_act=None, id_sact=None, id_hoi=None, level=None):
    assert sum([x is not None for x in [id_act, id_sact, id_hoi]]) == 1
    assert level in ['act', 'sact', 'hoi']

    if id_hoi is not None:
      assert level != 'hoi'

      if level == 'sact':
        id_sact = self.id_hoi_to_id_sact[id_hoi]
        return id_sact
      elif level == 'act':
        id_sact = self.id_hoi_to_id_sact[id_hoi]
        id_act = self.id_sact_to_id_act[id_sact]
        return id_act

    elif id_sact is not None:
      assert level != 'sact'

      if level == 'act':
        id_act = self.id_sact_to_id_act[id_sact]
        return id_act
      elif level == 'hoi':
        ids_hoi = self.id_hoi_to_id_sact.inverse[id_sact]
        return ids_hoi

    elif id_act is not None:
      assert level != 'act'

      if level == 'sact':
        ids_sact = self.id_sact_to_id_act.inverse[id_act]
        return ids_sact
      elif level == 'hoi':
        ids_hoi = itertools.chain(*[self.id_hoi_to_id_sact.inverse[id_sact]
                                    for id_sact in self.id_sact_to_id_act.inverse[id_act]])
        return ids_hoi

    raise ValueError
