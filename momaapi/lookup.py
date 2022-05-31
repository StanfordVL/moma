import itertools
import json
import os
import os.path as osp
import pickle

from .data import Bidict, lazydict, Metadatum, Act, SAct, HOI

"""
The Lookup class implements the following lookups:
 - split -> ids_act (one-to-many): retrieve(kind='id_act', key=split)
 - id_act -> ann_act, metadatum (one-to-one): retrieve(kind='ann_act' or 'metadatum', key=id_act)
 - id_sact -> ann_sact (one-to-one): retrieve(kind='ann_sact', key=id_sact)
 - id_hoi -> ann_hoi, window (one-to-one): retrieve(kind='ann_hoi' or 'window', key=id_hoi)

These keys can be traced across the MOMA hierarchy:
 - id_act -> ids_sact (one-to-many): trace(id_act=id_act, kind='sact')
 - id_act -> ids_hoi (one-to-many): trace(id_act=id_act, kind='hoi')
 - id_sact -> id_act (one-to-one): trace(id_sact=id_sact, kind='act')
 - id_sact -> ids_hoi (one-to-many): trace(id_sact=id_sact, kind='hoi')
 - id_hoi -> id_sact (one-to-one): trace(id_hoi=id_hoi, kind='sact')
 - id_hoi -> id_act (one-to-one): trace(id_hoi=id_hoi, kind='act')
 
The following variables can be mapped:
 - cid_fs+split <-> cid_std: get_cid()
"""


class Lookup:
  def __init__(self, dir_moma, taxonomy, paradigm, load_val):
    self.dir_moma = dir_moma
    self.taxonomy = taxonomy
    self.paradigm = paradigm
    self.load_val = load_val

    self.split_to_ids_act = None
    self.id_act_to_metadatum = None
    self.id_act_to_ann_act = None
    self.id_sact_to_ann_sact = None
    self.id_hoi_to_ann_hoi = None
    self.id_sact_to_id_act = None
    self.id_hoi_to_id_sact = None
    self.id_hoi_to_window = None

    self.read_splits()
    self.read_anns()

  def save_cache(self, id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi,
                 id_sact_to_id_act, id_hoi_to_id_sact, id_hoi_to_window):
    print('Lookup: save cache')
    named_variables = {
      'id_act_to_metadatum': id_act_to_metadatum,
      'id_act_to_ann_act': id_act_to_ann_act,
      'id_sact_to_ann_sact': id_sact_to_ann_sact,
      'id_hoi_to_ann_hoi': id_hoi_to_ann_hoi,
      'id_sact_to_id_act': id_sact_to_id_act,
      'id_hoi_to_id_sact': id_hoi_to_id_sact,
      'id_hoi_to_window': id_hoi_to_window
    }

    os.makedirs(os.path.join(self.dir_moma, f'anns/cache/{self.paradigm}/id_hoi_to_ann_hoi'), exist_ok=True)

    for name, variable in named_variables.items():
      assert variable is not None

      if name == 'id_hoi_to_ann_hoi':
        for id_hoi, ann_hoi in variable.items():
          with open(osp.join(self.dir_moma, f'anns/cache/id_hoi_to_ann_hoi', id_hoi), 'wb') as f:
            pickle.dump(ann_hoi, f)

      else:
        with open(osp.join(self.dir_moma, f'anns/cache', name), 'wb') as f:
          pickle.dump(variable, f)

  def load_cache(self):
    print('Lookup: load cache')
    variables = []
    for name in ['id_act_to_metadatum', 'id_act_to_ann_act', 'id_sact_to_ann_sact',
                 'id_sact_to_id_act', 'id_hoi_to_id_sact', 'id_hoi_to_clip']:
      with open(osp.join(self.dir_moma, f'anns/cache/', name), 'rb') as f:
        variable = pickle.load(f)
      variables.append(variable)

    ids_hoi = variables[4].keys()
    id_hoi_to_ann_hoi = lazydict(ids_hoi, osp.join(self.dir_moma, f'anns/cache/id_hoi_to_ann_hoi'))
    variables.insert(3, id_hoi_to_ann_hoi)

    return variables

  def get_cid(self, paradigm, split=None, cid_act=None, cid_sact=None):
    assert sum([x is not None for x in [cid_act, cid_sact]]) == 1
    if cid_act is not None:
      kind = 'act'
      cid_src = cid_act
    elif cid_sact is not None:
      kind = 'sact'
      cid_src = cid_sact
    else:
      raise ValueError

    if paradigm == 'standard':
      assert split is not None
      cname = self.taxonomy['few_shot'][kind][split][cid_src]
      cid_trg = self.taxonomy[kind].index(cname)
    elif paradigm == 'few-shot':
      cname = self.taxonomy[kind][cid_src]
      split = self.taxonomy['few_shot'][kind].inverse[cname]
      cid_trg = self.taxonomy['few_shot'][kind][split].index(cname)
    else:
      raise ValueError

    return cid_trg

  def read_anns(self):
    try:
      id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi, \
      id_sact_to_id_act, id_hoi_to_id_sact, id_hoi_to_window = self.load_cache()

    except FileNotFoundError:
      with open(osp.join(self.dir_moma, f'anns/anns.json'), 'r') as f:
        anns_raw = json.load(f)
      with open(osp.join(self.dir_moma, f'videos/interaction_frames/timestamps.json'), 'r') as f:
        timestamps = json.load(f)

      id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi = {}, {}, {}, {}
      id_sact_to_id_act, id_hoi_to_id_sact = {}, {}

      for ann_raw in anns_raw:
        ann_act_raw = ann_raw['activity']
        id_act_to_metadatum[ann_act_raw['id']] = Metadatum(ann_raw)
        cid_act = self.taxonomy['act'].index(ann_act_raw['class_name'])
        cid_act = self.get_cid(self.paradigm, cid_act=cid_act) if self.paradigm == 'few-shot' else cid_act
        id_act_to_ann_act[ann_act_raw['id']] = Act(ann_act_raw, cid_act)
        anns_sact_raw = ann_act_raw['sub_activities']

        for ann_sact_raw in anns_sact_raw:
          cid_sact = self.taxonomy['sact'].index(ann_sact_raw['class_name'])
          cid_sact = self.get_cid(self.paradigm, cid_sact=cid_sact) if self.paradigm == 'few-shot' else cid_sact
          id_sact_to_ann_sact[ann_sact_raw['id']] = SAct(ann_sact_raw, cid_sact)
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

      self.save_cache(id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi,
                      id_sact_to_id_act, id_hoi_to_id_sact, id_hoi_to_window)

    id_sact_to_id_act = Bidict(id_sact_to_id_act)
    id_hoi_to_id_sact = Bidict(id_hoi_to_id_sact)

    self.id_act_to_metadatum = id_act_to_metadatum
    self.id_act_to_ann_act = id_act_to_ann_act
    self.id_sact_to_ann_sact = id_sact_to_ann_sact
    self.id_hoi_to_ann_hoi = id_hoi_to_ann_hoi
    self.id_sact_to_id_act = id_sact_to_id_act
    self.id_hoi_to_id_sact = id_hoi_to_id_sact
    self.id_hoi_to_window = id_hoi_to_window

  def read_splits(self):
    # load split
    suffixes = {'standard': 'std', 'few-shot': 'fs'}
    path_split = osp.join(self.dir_moma, f'anns/split_{suffixes[self.paradigm]}.json')
    assert osp.isfile(path_split), f'Dataset split file does not exist: {path_split}'
    with open(path_split, 'r') as f:
      ids_act_splits = json.load(f)

    ids_act_train, ids_act_val, ids_act_test = ids_act_splits['train'], ids_act_splits['val'], ids_act_splits['test']

    if self.load_val:
      self.split_to_ids_act = {'train': ids_act_train, 'val': ids_act_val, 'test': ids_act_test}
    else:
      self.split_to_ids_act = {'train': ids_act_train+ids_act_val, 'test': ids_act_test}

  def retrieve(self, kind, key=None):
    if key is None:
      assert kind in ['splits', 'ids_act', 'ids_sact', 'ids_hoi',
                      'anns_act', 'metadata', 'anns_sact', 'anns_hoi', 'windows']

      if kind == 'splits':
        return self.split_to_ids_act.keys()
      elif kind == 'ids_act':
        return self.id_act_to_ann_act.keys()
      elif kind == 'ids_sact':
        return self.id_sact_to_ann_sact.keys()
      elif kind == 'ids_hoi':
        return self.id_hoi_to_ann_hoi.keys()
      elif kind == 'anns_act':
        return self.id_act_to_ann_act.values()
      elif kind == 'metadata':
        return self.id_act_to_metadatum.values()
      elif kind == 'anns_sact':
        return self.id_sact_to_ann_sact.values()
      elif kind == 'anns_hoi':
        return self.id_hoi_to_ann_hoi.values()
      elif kind == 'windows':
        return self.id_hoi_to_window.values()

    else:
      assert kind in ['ids_act', 'ann_act', 'metadatum', 'ann_sact', 'ann_hoi', 'window']

      if kind == 'ids_act':
        return self.split_to_ids_act[key]
      elif kind == 'ann_act':
        return self.id_act_to_ann_act[key]
      elif kind == 'metadatum':
        return self.id_act_to_metadatum[key]
      elif kind == 'ann_sact':
        return self.id_sact_to_ann_sact[key]
      elif kind == 'ann_hoi':
        return self.id_hoi_to_ann_hoi[key]
      elif kind == 'window':
        return self.id_hoi_to_window[key]

    raise ValueError(f'retrieve(kind={kind}, key={key})')

  def trace(self, kind, id_act=None, id_sact=None, id_hoi=None):
    assert sum([x is not None for x in [id_act, id_sact, id_hoi]]) == 1
    assert kind in ['id_act', 'id_sact', 'ids_sact', 'id_hoi', 'ids_hoi']

    if id_hoi is not None:
      assert kind in ['id_act', 'id_sact']

      if kind == 'id_sact':
        id_sact = self.id_hoi_to_id_sact[id_hoi]
        return id_sact
      elif kind == 'id_act':
        id_sact = self.id_hoi_to_id_sact[id_hoi]
        id_act = self.id_sact_to_id_act[id_sact]
        return id_act

    elif id_sact is not None:
      assert kind in ['id_act', 'ids_hoi']

      if kind == 'id_act':
        id_act = self.id_sact_to_id_act[id_sact]
        return id_act
      elif kind == 'ids_hoi':
        ids_hoi = self.id_hoi_to_id_sact.inverse[id_sact]
        return ids_hoi

    elif id_act is not None:
      assert kind in ['ids_sact', 'ids_hoi']

      if kind == 'ids_sact':
        ids_sact = self.id_sact_to_id_act.inverse[id_act]
        return ids_sact
      elif kind == 'ids_hoi':
        ids_hoi = itertools.chain(*[self.id_hoi_to_id_sact.inverse[id_sact]
                                    for id_sact in self.id_sact_to_id_act.inverse[id_act]])
        return ids_hoi

    raise ValueError
