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

The Lookup class implements the following lookups:
 - split -> ids_act (one-to-many): retrieve(kind='id_act', key=split)
 - id_act -> ann_act, metadatum (one-to-one): retrieve(kind='ann_act' or 'metadatum', key=id_act)
 - id_sact -> ann_sact (one-to-one): retrieve(kind='ann_sact', key=id_sact)
 - id_hoi -> ann_hoi, clip (one-to-one): retrieve(kind='ann_hoi' or 'clip', key=id_hoi)

These keys can be mapped across the MOMA hierarchy:
 - id_act -> ids_sact (one-to-many): map_id(id_act=id_act, kind='sact')
 - id_act -> ids_hoi (one-to-many): map_id(id_act=id_act, kind='hoi')
 - id_sact -> id_act (one-to-one): map_id(id_sact=id_sact, kind='act')
 - id_sact -> ids_hoi (one-to-many): map_id(id_sact=id_sact, kind='hoi')
 - id_hoi -> id_sact (one-to-one): map_id(id_hoi=id_hoi, kind='sact')
 - id_hoi -> id_act (one-to-one): map_id(id_hoi=id_hoi, kind='act')
 
Mapping activity and sub-activity class IDs between few-shot and standard paradigms:
 - cid_fs -> cid_std: map_cid(split=split, cid_act=cid_fs or cid_sact=cid_fs)
 - cid_std -> cid_fs: map_cid(split=split, cid_act=cid_std or cid_sact=cid_std)
"""


class Lookup:
  def __init__(self, dir_moma, taxonomy, reset_cache):
    self.taxonomy = taxonomy

    self.id_act_to_metadatum = None
    self.id_act_to_ann_act = None
    self.id_sact_to_ann_sact = None
    self.id_hoi_to_ann_hoi = None
    self.id_hoi_to_clip = None
    self.id_sact_to_id_act = None
    self.id_hoi_to_id_sact = None
    self.paradigm_and_split_to_ids_act = None

    self.__read_anns(dir_moma, reset_cache)
    self.__read_paradigms_and_splits(dir_moma)

  @staticmethod
  def __save_cache(dir_moma, id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi,
                   id_hoi_to_clip, id_sact_to_id_act, id_hoi_to_id_sact):
    dir_lookup = osp.join(dir_moma, 'anns/cache/lookup')
    os.makedirs(osp.join(dir_lookup, 'id_hoi'), exist_ok=True)

    named_variables = {
      'id_act_to_metadatum': id_act_to_metadatum,
      'id_act_to_ann_act': id_act_to_ann_act,
      'id_sact_to_ann_sact': id_sact_to_ann_sact,
      'id_hoi_to_ann_hoi': id_hoi_to_ann_hoi,
      'id_hoi_to_clip': id_hoi_to_clip,
      'id_sact_to_id_act': id_sact_to_id_act,
      'id_hoi_to_id_sact': id_hoi_to_id_sact,
    }

    for name, variable in named_variables.items():
      assert variable is not None

      if name == 'id_hoi_to_ann_hoi' or name == 'id_hoi_to_clip':
        for id_hoi, value in variable.items():
          with open(osp.join(dir_lookup, f'{name.replace("_to_", "/")}_{id_hoi}'), 'wb') as f:
            pickle.dump(value, f)

      else:
        with open(osp.join(dir_lookup, name), 'wb') as f:
          pickle.dump(variable, f)

    print('Lookup: save cache')

  @staticmethod
  def __load_cache(dir_moma):
    dir_lookup = osp.join(dir_moma, 'anns/cache/lookup')

    variables = []
    for name in ['id_act_to_metadatum', 'id_act_to_ann_act', 'id_sact_to_ann_sact',
                 'id_sact_to_id_act', 'id_hoi_to_id_sact']:
      with open(osp.join(dir_lookup, name), 'rb') as f:
        variable = pickle.load(f)
      variables.append(variable)

    id_hoi_to_ann_hoi = LazyDict(osp.join(dir_lookup, 'id_hoi'), 'ann_hoi')
    id_hoi_to_clip = LazyDict(osp.join(dir_lookup, 'id_hoi'), 'clip')
    variables.insert(3, id_hoi_to_ann_hoi)
    variables.insert(4, id_hoi_to_clip)

    assert len(id_hoi_to_ann_hoi.keys()) == len(variables[-1].keys()), \
        f'{len(id_hoi_to_ann_hoi.keys())} vs {len(variables[-1].keys())}'
    print('Lookup: load cache')
    return variables

  def __read_anns(self, dir_moma, reset_cache):
    dir_lookup = osp.join(dir_moma, 'anns/cache/lookup')
    if reset_cache and osp.exists(dir_lookup):
      shutil.rmtree(dir_lookup)

    try:
      id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi, id_hoi_to_clip, \
      id_sact_to_id_act, id_hoi_to_id_sact = self.__load_cache(dir_moma)

    except FileNotFoundError:
      with open(osp.join(dir_moma, f'anns/anns.json'), 'r') as f:
        anns_raw = json.load(f)
      with open(osp.join(dir_moma, f'anns/clips.json'), 'r') as f:
        info_clips = json.load(f)

      id_act_to_metadatum, id_act_to_ann_act = {}, {}
      id_sact_to_ann_sact = {}
      id_hoi_to_ann_hoi, id_hoi_to_clip = {}, {}
      id_sact_to_id_act, id_hoi_to_id_sact = {}, {}

      for ann_raw in anns_raw:
        ann_act_raw = ann_raw['activity']
        id_act_to_metadatum[ann_act_raw['id']] = Metadatum(ann_raw)
        id_act_to_ann_act[ann_act_raw['id']] = Act(ann_act_raw, self.taxonomy['act'])
        anns_sact_raw = ann_act_raw['sub_activities']

        for ann_sact_raw in anns_sact_raw:
          id_sact_to_ann_sact[ann_sact_raw['id']] = SAct(ann_sact_raw, self.taxonomy['sact'], 
                                                         self.taxonomy['actor'], self.taxonomy['object'])
          id_sact_to_id_act[ann_sact_raw['id']] = ann_act_raw['id']
          anns_hoi_raw = ann_sact_raw['higher_order_interactions']

          for ann_hoi_raw in anns_hoi_raw:
            id_hoi_to_ann_hoi[ann_hoi_raw['id']] = HOI(ann_hoi_raw,
                                                       self.taxonomy['actor'], self.taxonomy['object'],
                                                       self.taxonomy['ia'], self.taxonomy['ta'],
                                                       self.taxonomy['att'], self.taxonomy['rel'])
            if ann_hoi_raw['id'] in info_clips:  # Currently, only clips from the test set have been generated
              id_hoi_to_clip[ann_hoi_raw['id']] = Clip(ann_hoi_raw, info_clips[ann_hoi_raw['id']])
            id_hoi_to_id_sact[ann_hoi_raw['id']] = ann_sact_raw['id']

      self.__save_cache(dir_moma, id_act_to_metadatum, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi,
                        id_hoi_to_clip, id_sact_to_id_act, id_hoi_to_id_sact)

    id_sact_to_id_act = Bidict(id_sact_to_id_act)
    id_hoi_to_id_sact = Bidict(id_hoi_to_id_sact)

    self.id_act_to_metadatum = id_act_to_metadatum
    self.id_act_to_ann_act = id_act_to_ann_act
    self.id_sact_to_ann_sact = id_sact_to_ann_sact
    self.id_hoi_to_ann_hoi = id_hoi_to_ann_hoi
    self.id_sact_to_id_act = id_sact_to_id_act
    self.id_hoi_to_id_sact = id_hoi_to_id_sact
    self.id_hoi_to_clip = id_hoi_to_clip

  def __read_paradigms_and_splits(self, dir_moma):
    paradigms = ['standard', 'few-shot']
    splits = ['train', 'val', 'test']
    suffixes = {'standard': 'std', 'few-shot': 'fs'}

    paradigm_and_split_to_ids_act = {}
    for paradigm in paradigms:
      path_split = osp.join(dir_moma, f'anns/split_{suffixes[paradigm]}.json')
      assert osp.isfile(path_split), f'Dataset split file does not exist: {path_split}'
      with open(path_split, 'r') as f:
        ids_act = json.load(f)
      for split in splits:
        paradigm_and_split_to_ids_act[f'{paradigm}_{split}'] = ids_act[split]

    self.paradigm_and_split_to_ids_act = paradigm_and_split_to_ids_act

  def retrieve(self, kind, key=None):
    if key is None:
      assert kind in ['paradigms', 'splits', 'ids_act', 'ids_sact', 'ids_hoi',
                      'anns_act', 'metadata', 'anns_sact', 'anns_hoi', 'clips']

      if kind == 'paradigms':
        return [x.split('_')[0] for x in self.paradigm_and_split_to_ids_act.keys()]
      elif kind == 'splits':
        return [x.split('_')[1] for x in self.paradigm_and_split_to_ids_act.keys()]
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
      elif kind == 'clips':
        return self.id_hoi_to_clip.values()

    else:
      assert kind in ['ids_act', 'ann_act', 'metadatum', 'ann_sact', 'ann_hoi', 'clip']

      if kind == 'ids_act':
        return self.paradigm_and_split_to_ids_act[key]
      elif kind == 'ann_act':
        return self.id_act_to_ann_act[key]
      elif kind == 'metadatum':
        return self.id_act_to_metadatum[key]
      elif kind == 'ann_sact':
        return self.id_sact_to_ann_sact[key]
      elif kind == 'ann_hoi':
        return self.id_hoi_to_ann_hoi[key]
      elif kind == 'clip':
        return self.id_hoi_to_clip[key]

    raise ValueError(f'retrieve(kind={kind}, key={key})')

  def map_id(self, kind, id_act=None, id_sact=None, id_hoi=None):
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

  def map_cid(self, paradigm, split=None, cid_act=None, cid_sact=None):
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