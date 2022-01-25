import itertools
import json
import os

from .data import *


class MOMA:
  def __init__(self, dir_moma):
    self.dir_moma = dir_moma
    self.taxonomy = self.read_taxonomy()
    self.anns_act, self.anns_sact, self.anns_hoi, self.id_sact_to_act, self.id_hoi_to_sact = self.read_anns()

  def get_ids_act(self, cnames_act, ids_sact, ids_hoi):
    """
    same-level
     - cnames_act: get activity ids [ids_act] for given activity class names [cnames_act]
    bottom-up
     - ids_sact: get activity ids [ids_act] for given sub-activity ids [ids_sact]
     - ids_hoi: get activity ids [ids_act] for given higher-order interaction ids [ids_hoi]
    """
    assert not all(x is None for x in [cnames_act, ids_sact, ids_hoi])

    ids_act_all = []
    
    # cnames_act
    if cnames_act is not None:
      ids_act = []
      for ann_act in self.anns_act:
        cname_act = self.taxonomy['act'][ann_act.cid]
        if cname_act in cnames_act:
          ids_act.append(ann_act.id)
      ids_act_all.append(ids_act)

    # ids_sact
    if ids_sact is not None:
      ids_act = [self.id_sact_to_act[id_sact] for id_sact in ids_sact]
      ids_act_all.append(ids_act)

    # ids_hoi
    if ids_hoi is not None:
      ids_act = itertools.chain(*[self.id_sact_to_act[id_sact]
                                  for id_hoi in ids_hoi
                                  for id_sact in self.id_hoi_to_sact[id_hoi]])
      ids_act_all.append(ids_act)

    ids_act_all = list(set.intersection(*map(set, ids_act_all)))
    return ids_act_all

  def get_ids_sact(self,
                   cnames_sact=None, ids_act=None, ids_hoi=None,
                   cnames_actor=None, cnames_object=None,
                   cnames_ia=None, cnames_ta=None,
                   cnames_att=None, cnames_rel=None):
    """
    same-level
     - cnames_sact: get sub-activity ids [ids_sact] for given sub-activity class names [cnames_sact]
    top-down
     - ids_act: get sub-activity ids [ids_sact] for given activity ids [ids_act]
    bottom-up
     - ids_hoi: get sub-activity ids [ids_sact] for given higher-order interaction ids [ids_hoi]
     - cnames_actor: get sub-activity ids [ids_sact] for given actor class names [cnames_actor]
     - cnames_object: get sub-activity ids [ids_sact] for given object class names [cnames_object]
     - cnames_ia: get sub-activity ids [ids_sact] for given intransitive action class names [cnames_ia]
     - cnames_ta: get sub-activity ids [ids_sact] for given transitive action class names [cnames_ta]
     - cnames_att: get sub-activity ids [ids_sact] for given attribute class names [cnames_att]
     - cnames_rel: get sub-activity ids [ids_sact] for given relationship class names [cnames_rel]
    """
    assert not all(x is None for x in [cnames_sact, ids_act, ids_hoi,
                                       cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel])

    ids_sact_all = []

    # cnames_sact
    if cnames_sact is not None:
      ids_sact = []
      for ann_sact in self.anns_sact:
        cname_sact = self.taxonomy['sact'][ann_sact.cid]
        if cname_sact in cnames_sact:
          ids_sact.append(ann_sact.id)
      ids_sact_all.append(ids_sact)

    # ids_act
    if ids_act is not None:
      ids_sact = itertools.chain(*[self.id_sact_to_act.inverse[id_act] for id_act in ids_act])
      ids_sact_all.append(ids_sact)

    # ids_hoi
    if ids_hoi is not None:
      ids_sact = [self.id_hoi_to_sact[id_hoi] for id_hoi in ids_hoi]
      ids_sact_all.append(ids_sact)

    # cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel
    if not all(x is None for x in [cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel]):
      kwargs = {'cnames_actor': cnames_actor, 'cnames_object': cnames_object,
                'cnames_ia': cnames_ia, 'cnames_ta': cnames_ta,
                'cnames_att': cnames_att, 'cnames_rel': cnames_rel}
      ids_sact = [self.id_hoi_to_sact[id_hoi] for id_hoi in self.get_ids_hoi(**kwargs)]
      ids_sact_all.append(ids_sact)

    ids_sact_all = list(set.intersection(*map(set, ids_sact_all)))
    return ids_sact_all

  def get_ids_hoi(self,
                  ids_act=None, ids_sact=None,
                  cnames_actor=None, cnames_object=None,
                  cnames_ia=None, cnames_ta=None,
                  cnames_att=None, cnames_rel=None):
    """
    top-down
     - ids_act: get higher-order interaction ids [ids_hoi] for given activity ids [ids_act]
     - ids_sact: get higher-order interaction ids [ids_hoi] for given sub-activity ids [ids_sact]
    bottom-up
     - cnames_actor: get higher-order interaction ids [ids_hoi] for given actor class names [cnames_actor]
     - cnames_object: get higher-order interaction ids [ids_hoi] for given object class names [cnames_object]
     - cnames_ia: get higher-order interaction ids [ids_hoi] for given intransitive action class names [cnames_ia]
     - cnames_ta: get higher-order interaction ids [ids_hoi] for given transitive action class names [cnames_ta]
     - cnames_att: get higher-order interaction ids [ids_hoi] for given attribute class names [cnames_att]
     - cnames_rel: get higher-order interaction ids [ids_hoi] for given relationship class names [cnames_rel]
    """
    assert not all(x is None for x in [ids_act, ids_sact,
                                       cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel])

    ids_hoi_all = []

    # ids_act
    if ids_act is not None:
      ids_hoi = itertools.chain(*[self.id_hoi_to_sact.inverse[id_sact]
                                  for id_act in ids_act
                                  for id_sact in self.id_sact_to_act.inverse[id_act]])
      ids_hoi_all.append(ids_hoi)

    # ids_sact
    if ids_sact is not None:
      ids_hoi = itertools.chain(*[self.id_hoi_to_sact.inverse[id_sact] for id_sact in ids_sact])
      ids_hoi_all.append(ids_hoi)

    # cnames_actor, cnames_object, cnames_ia, cnames_ta, cnames_att, cnames_rel
    cnames_dict = {'actors': cnames_actor, 'objects': cnames_object,
                   'ias': cnames_ia, 'tas': cnames_ta,
                   'atts': cnames_att, 'rels': cnames_rel}
    for var, cnames in cnames_dict.items():
      if cnames is not None:
        ids_hoi = []
        for ann_hoi in self.anns_hoi:
          if not set(cnames).isdisjoint([x.cname for x in getattr(ann_hoi, var)]):
            ids_hoi.append(ann_hoi.id)
        ids_hoi_all.append(ids_hoi)

    ids_hoi_all = list(set.intersection(*map(set, ids_hoi_all)))
    return ids_hoi_all

  def get_id_sact(self, id_hoi):
    """
     - id_hoi: get sub-activity id [id_sact] for a given higher-order interaction id [id_hoi]
    """
    return self.id_hoi_to_sact[id_hoi]
    # self.id_sact_to_act, self.id_hoi_to_sact

  def get_id_act(self, id_sact=None, id_hoi=None):
    """
     - id_sact: get activity id [id_act] for a given sub-activity id [id_sact]
     - id_hoi: get activity id [id_act] for a given higher-order interaction id [id_hoi]
    """
    assert (id_sact is None) != (id_hoi is None)
    
    if id_sact is None:
      id_sact = self.id_hoi_to_sact[id_hoi]
    id_act = self.id_sact_to_act[id_sact]

    return id_act

  def get_ann_act(self, id_act):
    return self.anns_act[id_act]

  def get_ann_sact(self, id_sact):
    return self.anns_sact[id_sact]

  def get_ann_hoi(self, id_hoi):
    return self.anns_hoi[id_hoi]

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
      taxonomy_act_sact = bidict({cname_sact: cname_act for cname_act, cnames_sact in taxonomy_act_sact.items()
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
      'sact_to_act': taxonomy_act_sact
    }

    return taxonomy

  def read_anns(self):
    with open(os.path.join(self.dir_moma, 'anns/anns.json'), 'r') as f:
      anns_raw = json.load(f)

    anns_act, anns_sact, anns_hoi = {}, {}, {}
    id_sact_to_act, id_hoi_to_sact = {}, {}

    for ann_raw in anns_raw:
      ann_act_raw = ann_raw['activity']
      anns_act[ann_act_raw['id']] = Act(ann_act_raw, self.taxonomy['act'])
      anns_sact_raw = ann_act_raw['sub_activities']
      for ann_sact_raw in anns_sact_raw:
        anns_sact[ann_sact_raw['id']] = SAct(ann_sact_raw, self.taxonomy['sact'])
        anns_hoi_raw = ann_sact_raw['higher_order_interactions']
        id_sact_to_act[ann_sact_raw['id']] = ann_act_raw['id']
        for ann_hoi_raw in anns_hoi_raw:
          anns_hoi[ann_hoi_raw['id']] = HOI(ann_hoi_raw, self.taxonomy)
          id_hoi_to_sact[ann_hoi_raw['id']] = ann_sact_raw['id']

    id_sact_to_act = bidict(id_sact_to_act)
    id_hoi_to_sact = bidict(id_hoi_to_sact)

    return anns_act, anns_sact, anns_hoi, id_sact_to_act, id_hoi_to_sact
