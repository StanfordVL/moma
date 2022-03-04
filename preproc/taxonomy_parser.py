import csv
import json
import os
from pprint import pprint


class TaxonomyParser:
  def __init__(self, dir_moma):
    cn2en = {
      '人物类型': 'actor',
      '物体类型': 'object',
      '关系词': 'binary predicate',
      '元动作、互动': 'unary predicate'
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

    # taxonomy of attributes
    with open(os.path.join(dir_moma, 'anns/taxonomy/attribute.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row[:5] for row in reader][1:]
      taxonomy_att, cn2en_att = self.__get_taxonomy(rows)

    # taxonomy of relationships
    with open(os.path.join(dir_moma, 'anns/taxonomy/relationship.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row[:6] for row in reader][1:]
      taxonomy_rel, cn2en_rel = self.__get_taxonomy(rows)

    # taxonomy of intransitive actions
    with open(os.path.join(dir_moma, 'anns/taxonomy/intransitive_action.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row[:5] for row in reader][1:]
      taxonomy_ia, cn2en_ia = self.__get_taxonomy(rows)

    # taxonomy of transitive actions
    with open(os.path.join(dir_moma, 'anns/taxonomy/transitive_action.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row[:6] for row in reader][1:]
      taxonomy_ta, cn2en_ta = self.__get_taxonomy(rows)

    # taxonomy of activities and sub-activities
    with open(os.path.join(dir_moma, 'anns/taxonomy/act_sact.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row for row in reader][1:]
      taxonomy_act_sact, cn2en_act_sact = self.__get_taxonomy(rows)

    # translator
    cn2en = cn2en|cn2en_actor|cn2en_object|cn2en_ia|cn2en_ta|cn2en_att|cn2en_rel|cn2en_act_sact

    # LVIS mapper
    with open(os.path.join(dir_moma, 'anns/taxonomy/lvis.csv')) as f:
      reader = csv.reader(f, delimiter=',')
      rows = [row[:2] for row in reader][1:]
      lvis_mapper = {row[0]:int(row[1]) for row in rows}

    self.dir_moma = dir_moma
    self.taxonomy_actor = taxonomy_actor
    self.taxonomy_object = taxonomy_object
    self.taxonomy_ia = taxonomy_ia
    self.taxonomy_ta = taxonomy_ta
    self.taxonomy_att = taxonomy_att
    self.taxonomy_rel = taxonomy_rel
    self.taxonomy_act_sact = taxonomy_act_sact
    self.cn2en = cn2en
    self.lvis_mapper = lvis_mapper

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
      json.dump(self.taxonomy_actor, f, indent=2, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/object.json'), 'w') as f:
      json.dump(self.taxonomy_object, f, indent=2, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/intransitive_action.json'), 'w') as f:
      json.dump(self.taxonomy_ia, f, indent=2, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/transitive_action.json'), 'w') as f:
      json.dump(self.taxonomy_ta, f, indent=2, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/attribute.json'), 'w') as f:
      json.dump(self.taxonomy_att, f, indent=2, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/relationship.json'), 'w') as f:
      json.dump(self.taxonomy_rel, f, indent=2, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/act_sact.json'), 'w') as f:
      json.dump(self.taxonomy_act_sact, f, indent=2, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/cn2en.json'), 'w') as f:
      json.dump(self.cn2en, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/lvis.json'), 'w') as f:
      json.dump(self.lvis_mapper, f, ensure_ascii=False, indent=2, sort_keys=True)

  @staticmethod
  def __get_taxonomy(rows):
    taxonomy, cn2en, key, value = {}, {}, '', []

    for i in range(len(rows)):
      # new superclass or activity
      if rows[i][0] != '':
        key = rows[i][0]
        value = []
        cn2en[rows[i][1]] = rows[i][0]

      if rows[i][2] != 'REMOVE':
        if len(rows[i]) == 4:
          value.append(rows[i][2])
        else:  # predicate
          assert len(rows[i]) == 5 or len(rows[i]) == 6
          value.append(tuple([rows[i][2]]+rows[i][4:]))
      cn2en[rows[i][3]] = rows[i][2]

      # end of superclass or activity
      if i == len(rows)-1 or rows[i+1][0] != '':
        taxonomy[key] = sorted(set(value))

    return taxonomy, cn2en
