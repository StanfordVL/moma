import json
import os


class MOMA:
  def __init__(self, dir_moma):
    self.dir_moma = dir_moma
    self.taxonomies = self.read_taxonomies()
    self.anns = self.read_anns()

  def read_taxonomies(self):
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/actor.json'), 'r') as f:
      taxonomy_actor = json.load(f)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/object.json'), 'r') as f:
      taxonomy_object = json.load(f)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/intransitive_action.json'), 'r') as f:
      taxonomy_ia = json.load(f)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/transitive_action.json'), 'r') as f:
      taxonomy_ta = json.load(f)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/attribute.json'), 'r') as f:
      taxonomy_att = json.load(f)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/relationship.json'), 'r') as f:
      taxonomy_rel = json.load(f)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/act_sact.json'), 'r') as f:
      taxonomy_act_sact = json.load(f)

    taxonomies = {
      'actor': taxonomy_actor,
      'object': taxonomy_object,
      'intransitive_action': taxonomy_ia,
      'transitive_action': taxonomy_ta,
      'attribute': taxonomy_att,
      'relationship': taxonomy_rel,
      'sub_activity': taxonomy_act_sact
    }

    return taxonomies

  def read_anns(self):
    with open(os.path.join(self.dir_moma, 'anns/anns.json'), 'r') as f:
      anns = json.load(f)

    return anns
