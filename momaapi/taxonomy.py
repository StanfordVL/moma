import itertools
import json
import os

from .data import bidict


class Taxonomy(dict):
  def __init__(self, dir_moma):
    super().__init__()
    self.dir_moma = dir_moma
    self.taxonomy = self.read_taxonomy()

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
      lvis = json.load(f)
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

    taxonomy_fs = {
      'act_train': taxonomy_act_train,
      'act_val': taxonomy_act_val,
      'act_test': taxonomy_act_test,
      'sact_train': taxonomy_sact_train,
      'sact_val': taxonomy_sact_val,
      'sact_test': taxonomy_sact_test
    }

    taxonomy = {
      'actor': taxonomy_actor,
      'object': taxonomy_object,
      'ia': taxonomy_ia,
      'ta': taxonomy_ta,
      'att': taxonomy_att,
      'rel': taxonomy_rel,
      'act': taxonomy_act,
      'sact': taxonomy_sact,
      'sact_to_act': taxonomy_sact_to_act,
      'fs': taxonomy_fs,
      'lvis': lvis
    }

    return taxonomy

  def keys(self):
    return self.taxonomy.keys()

  def __getitem__(self, key):
    return self.taxonomy[key]

  def __len__(self):
    return len(self.taxonomy.keys())

  def __repr__(self):
    return repr(self.taxonomy)
