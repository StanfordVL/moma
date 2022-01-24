import csv
import json
import os
from pprint import pprint


class TaxonomyParser:
  def __init__(self, dir_moma):
    cn2en = {
      '人物类型': 'actor',
      '物体类型': 'object',
      '关系词': 'binary description',
      '元动作、互动': 'unary description'
    }

    # taxonomy of actors
    with open(os.path.join(dir_moma, 'anns/taxonomy/actor.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row[:4] for row in reader][1:]
      taxonomy_actor, cn2en_actor = self.__get_taxonomy(rows)

    # taxonomy of objects
    with open(os.path.join(dir_moma, 'anns/taxonomy/object.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row[:4] for row in reader][1:]
      taxonomy_object, cn2en_object = self.__get_taxonomy(rows)

    # taxonomy of atomic actions
    with open(os.path.join(dir_moma, 'anns/taxonomy/atomic_action.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row for row in reader][2:]
      rows_ia = [row[:5] for row in rows]
      rows_ta = [row[5:] for row in rows]

      taxonomy_ia, cn2en_ia = self.__get_taxonomy(rows_ia)
      taxonomy_ta, cn2en_ta = self.__get_taxonomy(rows_ta)

    # taxonomy of states
    with open(os.path.join(dir_moma, 'anns/taxonomy/state.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row for row in reader][2:]
      rows_att = [row[:5] for row in rows]
      rows_rel = [row[5:] for row in rows]

      taxonomy_att, cn2en_att = self.__get_taxonomy(rows_att)
      taxonomy_rel, cn2en_rel = self.__get_taxonomy(rows_rel)

    # taxonomy of activities and sub-activities
    with open(os.path.join(dir_moma, 'anns/taxonomy/act_sact.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row for row in reader][1:]
      taxonomy_act_sact, cn2en_act_sact = self.__get_taxonomy(rows)

    cn2en = cn2en|cn2en_actor|cn2en_object|cn2en_ia|cn2en_ta|cn2en_att|cn2en_rel|cn2en_act_sact

    self.dir_moma = dir_moma
    self.taxonomy_actor = taxonomy_actor
    self.taxonomy_object = taxonomy_object
    self.taxonomy_ia = taxonomy_ia
    self.taxonomy_ta = taxonomy_ta
    self.taxonomy_att = taxonomy_att
    self.taxonomy_rel = taxonomy_rel
    self.taxonomy_act_sact = taxonomy_act_sact
    self.cn2en = cn2en

  def dump(self, verbose=True):
    if verbose:
      print('Taxonomy of actors')
      pprint(self.taxonomy_actor)
      print('\n\nTaxonomy of objects')
      pprint(self.taxonomy_object)
      print('\n\nTaxonomy of intransitive actions')
      pprint(self.taxonomy_ia)
      print('\n\nTaxonomy of transitive actions')
      pprint(self.taxonomy_ta)
      print('\n\nTaxonomy of attributes')
      pprint(self.taxonomy_att)
      print('\n\nTaxonomy of relationships')
      pprint(self.taxonomy_rel)
      print('\n\nTaxonomy of activities and sub-activities')
      pprint(self.taxonomy_act_sact)
      print('\n\n Chinese to English dictionary')
      pprint(self.cn2en)

    with open(os.path.join(self.dir_moma, 'anns/taxonomy/actor.json'), 'w') as f:
      json.dump(self.taxonomy_actor, f, indent=4, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/object.json'), 'w') as f:
      json.dump(self.taxonomy_object, f, indent=4, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/intransitive_action.json'), 'w') as f:
      json.dump(self.taxonomy_ia, f, indent=4, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/transitive_action.json'), 'w') as f:
      json.dump(self.taxonomy_ta, f, indent=4, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/attribute.json'), 'w') as f:
      json.dump(self.taxonomy_att, f, indent=4, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/relationship.json'), 'w') as f:
      json.dump(self.taxonomy_rel, f, indent=4, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/act_sact.json'), 'w') as f:
      json.dump(self.taxonomy_act_sact, f, indent=4, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/cn2en.json'), 'w') as f:
      json.dump(self.cn2en, f, ensure_ascii=False, indent=4, sort_keys=True)

  @staticmethod
  def __get_taxonomy(rows):
    taxonomy, cn2en, key, value = {}, {}, '', []

    for i in range(len(rows)):
      if rows[i][0] != '':
        key = rows[i][0]
        value = []
        cn2en[rows[i][1]] = rows[i][0]

      if len(rows[i]) == 4:  # entity
        value.append(rows[i][2])
      else:  # description
        assert len(rows[i]) == 5 or len(rows[i]) == 6
        value.append(tuple([rows[i][2]]+rows[i][4:]))
      cn2en[rows[i][3]] = rows[i][2]

      if i == len(rows)-1 or rows[i+1][0] != '' or all([col == '' for col in rows[i+1]]):
        taxonomy[key] = sorted(value)
      if i < len(rows)-1 and all([col == '' for col in rows[i+1]]):
        break

    return taxonomy, cn2en
