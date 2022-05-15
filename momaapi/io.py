import itertools
import json
import os
import pickle

from .data import *


def save_cache(dir_cache, named_variables):
  os.makedirs(dir_cache, exist_ok=True)

  for name, variable in named_variables.items():
    assert variable is not None
    with open(os.path.join(dir_cache, name), 'wb') as f:
      pickle.dump(variable, f)


def load_cache(dir_cache, names):
  if not all([os.path.exists(os.path.join(dir_cache, name)) for name in names]):
    raise FileNotFoundError

  variables = []
  for name in names:
    with open(os.path.join(dir_cache, name), 'rb') as f:
      variable = pickle.load(f)
      variables.append(variable)

  return variables


class IO:
  def __init__(self, dir_moma):
    self.dir_moma = dir_moma
    self.taxonomy = None

  def read_taxonomy(self):
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
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/lvis.json'), 'r') as f:
      lvis_mapper = json.load(f)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/few_shot.json'), 'r') as f:
      taxonomy_fs = json.load(f)
      taxonomy_act_train = sorted(taxonomy_fs['train'])
      taxonomy_act_val = sorted(taxonomy_fs['val'])
      taxonomy_act_test = sorted(taxonomy_fs['test'])
      taxonomy_sact_train = [list(taxonomy_sact_to_act.inverse[x]) for x in taxonomy_act_train]
      taxonomy_sact_val = [list(taxonomy_sact_to_act.inverse[x]) for x in taxonomy_act_val]
      taxonomy_sact_test = [list(taxonomy_sact_to_act.inverse[x]) for x in taxonomy_act_test]
      taxonomy_sact_train = sorted(itertools.chain(*taxonomy_sact_train))
      taxonomy_sact_val = sorted(itertools.chain(*taxonomy_sact_val))
      taxonomy_sact_test = sorted(itertools.chain(*taxonomy_sact_test))

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

    taxonomy_fs = {
      'act_train': taxonomy_act_train,
      'act_val': taxonomy_act_val,
      'act_test': taxonomy_act_test,
      'sact_train': taxonomy_sact_train,
      'sact_val': taxonomy_sact_val,
      'sact_test': taxonomy_sact_test
    }

    self.taxonomy = taxonomy
    return taxonomy, taxonomy_fs, lvis_mapper

  def read_anns(self):
    if self.taxonomy is None:
      raise Exception('read_taxonomy() should be called first.')

    dir_cache = os.path.join(self.dir_moma, 'anns/cache')

    try:
      names = ['metadata', 'id_act_to_ann_act', 'id_sact_to_ann_sact', 'id_hoi_to_ann_hoi', 'id_sact_to_id_act',
               'id_hoi_to_id_sact']
      metadata, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi, id_sact_to_id_act, id_hoi_to_id_sact = \
          load_cache(dir_cache, names)

    except FileNotFoundError:
      with open(os.path.join(self.dir_moma, f'anns/anns.json'), 'r') as f:
        anns_raw = json.load(f)

      metadata, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi = {}, {}, {}, {}
      id_sact_to_id_act, id_hoi_to_id_sact = {}, {}

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

      named_variables = {
        'metadata': metadata,
        'id_act_to_ann_act': id_act_to_ann_act,
        'id_sact_to_ann_sact': id_sact_to_ann_sact,
        'id_hoi_to_ann_hoi': id_hoi_to_ann_hoi,
        'id_sact_to_id_act': id_sact_to_id_act,
        'id_hoi_to_id_sact': id_hoi_to_id_sact
      }
      save_cache(dir_cache, named_variables)

    id_sact_to_id_act = bidict(id_sact_to_id_act)
    id_hoi_to_id_sact = bidict(id_hoi_to_id_sact)

    with open(os.path.join(self.dir_moma, f'videos/interaction_frames/timestamps.json'), 'r') as f:
      windows = json.load(f)

    return metadata, id_act_to_ann_act, id_sact_to_ann_sact, id_hoi_to_ann_hoi, id_sact_to_id_act, id_hoi_to_id_sact, \
           windows

  def read_splits(self, few_shot, load_val):
    # load split
    path_split = os.path.join(self.dir_moma, 'anns/split_fs.json' if few_shot else 'anns/split.json')
    if not os.path.isfile(path_split):
      print(f'Dataset split file does not exist: {path_split}')
      return None, None
    with open(path_split, 'r') as f:
      ids_act_splits = json.load(f)

    ids_act_train, ids_act_val, ids_act_test = ids_act_splits['train'], ids_act_splits['val'], ids_act_splits['test']

    if load_val:
      return {'train': ids_act_train, 'val': ids_act_val, 'test': ids_act_test}
    else:
      return {'train': ids_act_train+ids_act_val, 'test': ids_act_test}
