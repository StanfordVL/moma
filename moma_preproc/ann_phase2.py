import itertools
import json
import math
import os

from .data import *
from .utils import *


class AnnPhase2:
  def __init__(self, dir_moma, fname_ann):
    anns_raw = []
    with open(os.path.join(dir_moma, 'anns', fname_ann), 'r') as fs:
      for f in fs:
        anns_raw.append(json.loads(f))
    anns_sact_raw = [list(v) for _, v in itertools.groupby(anns_raw, lambda x: x['id'])]
    anns_sact_raw = {self.get_id_sact(ann_sact_raw):ann_sact_raw for ann_sact_raw in anns_sact_raw}

    with open(os.path.join(dir_moma, 'anns/taxonomy/actor.json'), 'r') as f:
      taxonomy_actor = json.load(f)
      taxonomy_actor = sorted(itertools.chain(*[taxonomy_actor[key] for key in taxonomy_actor]))
    with open(os.path.join(dir_moma, 'anns/taxonomy/object.json'), 'r') as f:
      taxonomy_object = json.load(f)
      taxonomy_object = sorted(itertools.chain(*[taxonomy_object[key] for key in taxonomy_object]))
    with open(os.path.join(dir_moma, 'anns/taxonomy/intransitive_action.json'), 'r') as f:
      taxonomy_ia = json.load(f)
      taxonomy_ia = sorted(itertools.chain(*[taxonomy_ia[key] for key in taxonomy_ia]))
      taxonomy_ia = [tuple(x) for x in taxonomy_ia]
    with open(os.path.join(dir_moma, 'anns/taxonomy/transitive_action.json'), 'r') as f:
      taxonomy_ta = json.load(f)
      taxonomy_ta = sorted(itertools.chain(*[taxonomy_ta[key] for key in taxonomy_ta]))
      taxonomy_ta = [tuple(x) for x in taxonomy_ta]
    with open(os.path.join(dir_moma, 'anns/taxonomy/attribute.json'), 'r') as f:
      taxonomy_att = json.load(f)
      taxonomy_att = sorted(itertools.chain(*[taxonomy_att[key] for key in taxonomy_att]))
      taxonomy_att = [tuple(x) for x in taxonomy_att]
    with open(os.path.join(dir_moma, 'anns/taxonomy/relationship.json'), 'r') as f:
      taxonomy_rel = json.load(f)
      taxonomy_rel = sorted(itertools.chain(*[taxonomy_rel[key] for key in taxonomy_rel]))
      taxonomy_rel = [tuple(x) for x in taxonomy_rel]
    with open(os.path.join(dir_moma, 'anns/taxonomy/cn2en.json'), 'r') as f:
      cn2en = json.load(f)

    taxonomy_unary = taxonomy_ia+taxonomy_att
    taxonomy_binary = taxonomy_ta+taxonomy_rel

    self.anns_sact_raw = anns_sact_raw  # dict
    self.taxonomy_actor = taxonomy_actor
    self.taxonomy_object = taxonomy_object
    self.taxonomy_ia = taxonomy_ia
    self.taxonomy_ta = taxonomy_ta
    self.taxonomy_att = taxonomy_att
    self.taxonomy_rel = taxonomy_rel
    self.taxonomy_unary = taxonomy_unary
    self.taxonomy_binary = taxonomy_binary
    self.cn2en = cn2en
    self.ids_sact = []

    self.__fix()

  @staticmethod
  def __trim(ann_sact_raw, num_hois_trim, is_start):
    num_hois = len(ann_sact_raw)
    record = ann_sact_raw[0]['task']['task_params']['record']
    duration = record['metadata']['additionalInfo']['duration']-num_hois_trim
    id_hoi_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']

    for j in range(num_hois_trim):
      id_hoi_to_timestamp.pop(str(num_hois-j))

    for j in range(num_hois):
      ann_sact_raw[j]['task']['task_params']['record']['metadata']['additionalInfo']['duration'] = duration
      ann_sact_raw[j]['task']['task_params']['record']['metadata']['additionalInfo']['framesTimestamp'] = \
          id_hoi_to_timestamp

    if is_start:
      for j in range(num_hois-num_hois_trim):
        ann_sact_raw[num_hois-j-1]['task']['task_params']['record']['attachment'] = \
          ann_sact_raw[num_hois-num_hois_trim-j-1]['task']['task_params']['record']['attachment']
      ann_sact_raw = ann_sact_raw[num_hois_trim:]
    else:
      ann_sact_raw = ann_sact_raw[:-num_hois_trim]

    return ann_sact_raw

  def __fix(self):
    # '3361': start, 31 -> 33
    self.anns_sact_raw['3361'] = self.__trim(self.anns_sact_raw['3361'], 2, True)
    # '5730': start, 55 -> 56
    self.anns_sact_raw['5730'] = self.__trim(self.anns_sact_raw['5730'], 1, True)
    # '6239': end, 59 -> 49
    self.anns_sact_raw['6239'] = self.__trim(self.anns_sact_raw['6239'], 10, False)
    # '6679': start, 00 -> 05
    self.anns_sact_raw['6679'] = self.__trim(self.anns_sact_raw['6679'], 5, True)
    # '9534': end, 19 -> 09
    self.anns_sact_raw['9534'] = self.__trim(self.anns_sact_raw['9534'], 10, False)
    # '11065': end, 30 -> 20
    self.anns_sact_raw['11065'] = self.__trim(self.anns_sact_raw['11065'], 10, False)

    ids_sact_rm = ['27', '198', '199', '653', '1535', '1536', '3775', '4024', '5531', '5629',
                    '5729', '6178', '6478', '7073', '7074', '7076', '7350', '9713', '10926', '10927',
                    '11168', '11570', '12696', '12697', '15225', '15403', '15579', '15616']
    for iid_sact_rm in ids_sact_rm:
      if iid_sact_rm in self.anns_sact_raw:
        self.anns_sact_raw.pop(iid_sact_rm)

  @staticmethod
  def get_id_sact(ann_sact_raw):
    record = ann_sact_raw[0]['task']['task_params']['record']
    id_sact = record['attachment'].split('_')[-1][:-4].split('/')[0]
    return id_sact

  @staticmethod
  def get_id_hoi(ann_hoi_raw):
    record = ann_hoi_raw['task']['task_params']['record']
    id_sact, timestamp = record['attachment'].split('_')[-1][:-4].split('/')
    timestamp = float(timestamp)/1000000
    id_hoi_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']
    id_hoi = None
    for key in id_hoi_to_timestamp:
      if math.isclose(timestamp, id_hoi_to_timestamp[key], abs_tol=1e-6):
        id_hoi = key
    return id_hoi

  @staticmethod
  def get_timestamp(ann_hoi_raw):
    record = ann_hoi_raw['task']['task_params']['record']
    id_sact, timestamp = record['attachment'].split('_')[-1][:-4].split('/')
    timestamp = float(timestamp)/1000000
    return timestamp

  def __inspect_ann_sact(self, ann_sact_raw):
    # get id_sact, ids_hoi, and num_hois
    record = ann_sact_raw[0]['task']['task_params']['record']
    id_sact_real = record['attachment'].split('_')[-1][:-4].split('/')[0]
    id_hoi_to_timestamp_real = record['metadata']['additionalInfo']['framesTimestamp']
    num_hois_real = len(ann_sact_raw)
    ids_hoi_real = sorted(id_hoi_to_timestamp_real.keys(), key=int)
    assert ids_hoi_real[0] == '1' and ids_hoi_real[-1] == str(len(ids_hoi_real))

    errors = []
    anns_sact_actor, anns_sact_object = [], []
    for i, ann_hoi_raw in enumerate(ann_sact_raw):
      # actor
      anns_hoi_actor_raw = ann_hoi_raw['task_result']['annotations'][0]['slotsChildren']
      anns_hoi_actor = [Entity(ann_actor_raw, self.cn2en) for ann_actor_raw in anns_hoi_actor_raw]
      anns_sact_actor += anns_hoi_actor

      # object
      anns_hoi_object_raw = ann_hoi_raw['task_result']['annotations'][1]['slotsChildren']
      anns_hoi_object = [Entity(ann_object_raw, self.cn2en) for ann_object_raw in anns_hoi_object_raw]
      anns_sact_object += anns_hoi_object

      # check id_sact, ids_hoi, and num_hois
      record = ann_hoi_raw['task']['task_params']['record']

      id_sact, timestamp = record['attachment'].split('_')[-1][:-4].split('/')
      timestamp = float(timestamp)/1000000
      assert id_sact == id_sact_real

      id_hoi_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']
      id_hoi = None
      for key in id_hoi_to_timestamp:
        if math.isclose(timestamp, id_hoi_to_timestamp[key], abs_tol=1e-6):
          id_hoi = key
      assert id_hoi_to_timestamp == id_hoi_to_timestamp_real
      assert id_hoi is not None and id_hoi == str(i+1), f'{id_hoi} != {i+1}'

      num_hois = len(id_hoi_to_timestamp)
      assert num_hois == num_hois_real

      id_sact = record['metadata']['additionalInfo']['videoName'].split('_')[-1].split('/')[0]
      assert id_sact == id_sact_real

    ids_sact_actor = sort(set([ann_sact_actor.id for ann_sact_actor in anns_sact_actor]))
    ids_sact_object = sort(set([ann_sact_object.id for ann_sact_object in anns_sact_object]))
    anns_instances_actor = [list(v) for _, v in itertools.groupby(anns_sact_actor, lambda x: x.id)]
    anns_instances_object = [list(v) for _, v in itertools.groupby(anns_sact_object, lambda x: x.id)]

    # if not is_consecutive(ids_sact_actor):
    #   errors.append(f'[actor instance] ids not consecutive {ids_sact_actor}')

    # if not is_consecutive(ids_sact_object):
    #   errors.append(f'[object instance] ids not consecutive {ids_sact_object}')

    for anns_instance_actor in anns_instances_actor:
      cnames = [ann_instance_actor.cname for ann_instance_actor in anns_instance_actor]
      if len(set(cnames)) != 1:
        errors.append(f'[actor instance] id {anns_instance_actor[0].id} '
                      f'corresponds to more than one cname {set(cnames)}')

    for anns_instance_object in anns_instances_object:
      cnames = [ann_instance_object.cname for ann_instance_object in anns_instance_object]
      if len(set(cnames)) != 1:
        errors.append(f'[object instance] id {anns_instance_object[0].id} '
                      f'corresponds to more than one cname {set(cnames)}')

    return errors

  def __inspect_ann_hoi(self, ann_hoi_raw):
    errors = []

    assert len(ann_hoi_raw['task_result']['annotations']) == 4

    """ actor & object """
    ids = []
    for i, kind in enumerate(['actor', 'object']):
      assert self.cn2en[ann_hoi_raw['task_result']['annotations'][i]['label']] == kind
      anns_entity_raw = ann_hoi_raw['task_result']['annotations'][i]['slotsChildren']
      anns_entity = [Entity(ann_entity_raw, self.cn2en) for ann_entity_raw in anns_entity_raw]

      for ann_entity in anns_entity:
        # check kind
        assert ann_entity.kind == kind, f'[{kind}] wrong kind {ann_entity.kind}'

        # check cname
        taxonomy = self.taxonomy_actor if kind == 'actor' else self.taxonomy_object
        assert ann_entity.cname in taxonomy, f'[{kind}] unseen cname {ann_entity.cname}'

        # check id
        if not ((kind == 'actor' and is_actor(ann_entity.id)) or kind == 'object' and is_object(ann_entity.id)):
          errors.append(f'[{kind}] wrong id format {ann_entity.id}'.encode('unicode_escape').decode('utf-8'))

        # check bbox
        if ann_entity.bbox.x < 0 or ann_entity.bbox.y < 0 or ann_entity.bbox.width <= 0 or ann_entity.bbox.height <= 0:
          errors.append(f'[{kind}] wrong bbox size {ann_entity.bbox}')

      ids += [ann_entity.id for ann_entity in anns_entity]

    # check duplicate ids
    if len(set(ids)) != len(ids):
      errors.append(f'[actor/object] duplicate ids {ids}')

    """ binary description & unary description """
    for i, kind in enumerate(['binary description', 'unary description']):
      assert self.cn2en[ann_hoi_raw['task_result']['annotations'][i+2]['label']] == kind
      anns_description_raw = ann_hoi_raw['task_result']['annotations'][i+2]['slotsChildren']
      anns_description = [Description(ann_description_raw, self.cn2en)
                          for ann_description_raw in anns_description_raw]

      for ann_description in anns_description:
        # check kind
        if ann_description.kind != kind:
          errors.append(f'[{kind}] wrong kind {ann_description.kind}')

        # check cname
        taxonomy = self.taxonomy_binary if kind == 'binary description' else self.taxonomy_unary
        if ann_description.cname not in [x[0] for x in taxonomy]:
          errors.append(f'[{kind}] unseen cname {ann_description.cname}')

        # check ids_associated
        if kind == 'binary description':
          if ann_description.ids_associated[0] != '(' or \
             ann_description.ids_associated[-1] != ')' or \
             len(ann_description.ids_associated[1:-1].split('),(')) != 2:
            errors.append(f'[{kind}] wrong ids_associated format {ann_description.ids_associated}')
            continue

          ids_src = ann_description.ids_associated[1:-1].split('),(')[0].split(',')
          ids_trg = ann_description.ids_associated[1:-1].split('),(')[1].split(',')
          if not are_entities(ids_src+ids_trg):
            errors.append(f'[{kind}] wrong ids_associated format {ids_src} -> {ids_trg}')
            continue

          if not set(ids_src+ids_trg).issubset(ids):
            errors.append(f'[{kind}] unseen ids_associated {set(ids_src+ids_trg)} in {ids}')
            continue

          cnames_binary = [x[0] for x in self.taxonomy_binary]
          if ann_description.cname not in cnames_binary:  # unseen cname
            continue
          index = cnames_binary.index(ann_description.cname)
          kind_src, kind_trg = self.taxonomy_binary[index][1:]
          if (kind_src == 'actor' and not are_actors(ids_src)) or \
             (kind_src == 'object' and not are_objects(ids_src)) or \
             (kind_src == 'actor/object' and not are_entities(ids_src)) or \
             (kind_trg == 'actor' and not are_actors(ids_trg)) or \
             (kind_trg == 'object' and not are_objects(ids_trg)) or \
             (kind_trg == 'actor/object' and not are_entities(ids_trg)):
            errors.append(f'[{kind}] wrong ids_associated {ann_description.ids_associated} '
                          f'for kinds {kind_src} -> {kind_trg}')

        elif kind == 'unary description':
          ids_src = ann_description.ids_associated.split(',')
          if not are_actors(ids_src):
            errors.append(f'[{kind}] wrong ids_associated format {ann_description.ids_associated}')
            continue

          if not set(ids_src).issubset(ids):
            errors.append(f'[{kind}] unseen ids_associated {set(ids_src)} in {ids}')

          cnames_unary = [x[0] for x in self.taxonomy_unary]
          index = cnames_unary.index(ann_description.cname)
          kind_src = self.taxonomy_unary[index][1]
          if (kind_src == 'actor' and not are_actors(ids_src)) or \
             (kind_src == 'object' and not are_objects(ids_src)) or \
             (kind_src == 'actor/object' and not are_entities(ids_src)):
            errors.append(f'[{kind}] wrong ids_associated {ann_description.ids_associated} for kinds {kind_src}')

    return errors

  def inspect(self, verbose=True):
    errors = []
    for id_sact, ann_sact_raw in self.anns_sact_raw.items():
      errors_sact = self.__inspect_ann_sact(ann_sact_raw)
      if verbose and len(errors_sact) > 0:
        msg = errors_sact[0] if len(errors_sact) == 1 else '; '.join(errors_sact)
        print(f'Video {id_sact}; {msg}')

      errors_hoi = []
      for ann_hoi_raw in ann_sact_raw:
        id_hoi = self.get_id_hoi(ann_hoi_raw)
        errors_hoi += self.__inspect_ann_hoi(ann_hoi_raw)
        if verbose and len(errors_hoi) > 0:
          msg = errors_hoi[0] if len(errors_hoi) == 1 else '; '.join(errors_hoi)
          print(f'Video {id_sact} Image {id_hoi}; {msg}')

      # error-free sub-activities
      if len(errors_sact) == 0 and len(errors_hoi) == 0:
        self.ids_sact.append(id_sact)
      else:
        errors += errors_sact+errors_hoi

    print('\n ---------- REPORT (Phase 2) ----------')
    print(f'Number of error-free sub-activity instances: {len(self.anns_sact_raw)} -> {len(self.ids_sact)}')
    print(f'Number of errors: {len(errors)}')
