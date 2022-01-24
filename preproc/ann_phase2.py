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
    anns_sact_raw = {self.get_iid_sact(ann_sact_raw):ann_sact_raw for ann_sact_raw in anns_sact_raw}

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
    self.iids_sact = []

    self.__fix()

  @staticmethod
  def __trim(ann_sact_raw, num_hois_trim, is_start):
    num_hois = len(ann_sact_raw)
    record = ann_sact_raw[0]['task']['task_params']['record']
    duration = record['metadata']['additionalInfo']['duration']-num_hois_trim
    iid_hoi_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']

    for j in range(num_hois_trim):
      iid_hoi_to_timestamp.pop(str(num_hois-j))

    for j in range(num_hois):
      ann_sact_raw[j]['task']['task_params']['record']['metadata']['additionalInfo']['duration'] = duration
      ann_sact_raw[j]['task']['task_params']['record']['metadata']['additionalInfo']['framesTimestamp'] = \
          iid_hoi_to_timestamp

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

    iids_sact_rm = ['27', '198', '199', '653', '1535', '1536', '3775', '4024', '5531', '5629',
                    '5729', '6178', '6478', '7073', '7074', '7076', '7350', '9713', '10926', '10927',
                    '11168', '11570', '12696', '12697', '15225', '15403', '15579', '15616']
    for iiid_sact_rm in iids_sact_rm:
      if iiid_sact_rm in self.anns_sact_raw:
        self.anns_sact_raw.pop(iiid_sact_rm)

  @staticmethod
  def get_iid_sact(ann_sact_raw):
    record = ann_sact_raw[0]['task']['task_params']['record']
    iid_sact = record['attachment'].split('_')[-1][:-4].split('/')[0]
    return iid_sact

  @staticmethod
  def get_iid_hoi(ann_hoi_raw):
    record = ann_hoi_raw['task']['task_params']['record']
    iid_sact, timestamp = record['attachment'].split('_')[-1][:-4].split('/')
    timestamp = float(timestamp)/1000000
    iid_hoi_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']
    iid_hoi = None
    for key in iid_hoi_to_timestamp:
      if math.isclose(timestamp, iid_hoi_to_timestamp[key], abs_tol=1e-6):
        iid_hoi = key
    return iid_hoi

  @staticmethod
  def get_timestamp(ann_hoi_raw):
    record = ann_hoi_raw['task']['task_params']['record']
    iid_sact, timestamp = record['attachment'].split('_')[-1][:-4].split('/')
    timestamp = float(timestamp)/1000000
    return timestamp

  def __inspect_ann_sact(self, ann_sact_raw):
    # get iid_sact, ids_hoi, and num_hois
    record = ann_sact_raw[0]['task']['task_params']['record']
    iid_sact_real = record['attachment'].split('_')[-1][:-4].split('/')[0]
    iid_hoi_to_timestamp_real = record['metadata']['additionalInfo']['framesTimestamp']
    num_hois_real = len(ann_sact_raw)
    iids_hoi_real = sorted(iid_hoi_to_timestamp_real.keys(), key=int)
    assert iids_hoi_real[0] == '1' and iids_hoi_real[-1] == str(len(iids_hoi_real))

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

      # check iid_sact, ids_hoi, and num_hois
      record = ann_hoi_raw['task']['task_params']['record']

      iid_sact, timestamp = record['attachment'].split('_')[-1][:-4].split('/')
      timestamp = float(timestamp)/1000000
      assert iid_sact == iid_sact_real

      iid_hoi_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']
      iid_hoi = None
      for key in iid_hoi_to_timestamp:
        if math.isclose(timestamp, iid_hoi_to_timestamp[key], abs_tol=1e-6):
          iid_hoi = key
      assert iid_hoi_to_timestamp == iid_hoi_to_timestamp_real
      assert iid_hoi is not None and iid_hoi == str(i+1), f'{iid_hoi} != {i+1}'

      num_hois = len(iid_hoi_to_timestamp)
      assert num_hois == num_hois_real

      iid_sact = record['metadata']['additionalInfo']['videoName'].split('_')[-1].split('/')[0]
      assert iid_sact == iid_sact_real

    iids_sact_actor = sort(set([ann_sact_actor.iid for ann_sact_actor in anns_sact_actor]))
    iids_sact_object = sort(set([ann_sact_object.iid for ann_sact_object in anns_sact_object]))
    anns_instances_actor = [list(v) for _, v in itertools.groupby(anns_sact_actor, lambda x: x.iid)]
    anns_instances_object = [list(v) for _, v in itertools.groupby(anns_sact_object, lambda x: x.iid)]

    # if not is_consecutive(iids_sact_actor):
    #   errors.append(f'[actor instance] iids not consecutive {iids_sact_actor}')

    # if not is_consecutive(iids_sact_object):
    #   errors.append(f'[object instance] iids not consecutive {iids_sact_object}')

    for anns_instance_actor in anns_instances_actor:
      cnames = [ann_instance_actor.cname for ann_instance_actor in anns_instance_actor]
      if len(set(cnames)) != 1:
        errors.append(f'[actor instance] iid {anns_instance_actor[0].iid} '
                      f'corresponds to more than one cname {set(cnames)}')

    for anns_instance_object in anns_instances_object:
      cnames = [ann_instance_object.cname for ann_instance_object in anns_instance_object]
      if len(set(cnames)) != 1:
        errors.append(f'[object instance] iid {anns_instance_object[0].iid} '
                      f'corresponds to more than one cname {set(cnames)}')

    return errors

  def __inspect_ann_hoi(self, ann_hoi_raw):
    errors = []

    assert len(ann_hoi_raw['task_result']['annotations']) == 4

    """ actor & object """
    iids = []
    for i, type in enumerate(['actor', 'object']):
      assert self.cn2en[ann_hoi_raw['task_result']['annotations'][i]['label']] == type
      anns_entity_raw = ann_hoi_raw['task_result']['annotations'][i]['slotsChildren']
      anns_entity = [Entity(ann_entity_raw, self.cn2en) for ann_entity_raw in anns_entity_raw]

      for ann_entity in anns_entity:
        # check type
        assert ann_entity.type == type, f'[{type}] wrong type {ann_entity.type}'

        # check cname
        taxonomy = self.taxonomy_actor if type == 'actor' else self.taxonomy_object
        assert ann_entity.cname in taxonomy, f'[{type}] unseen cname {ann_entity.cname}'

        # check iid
        if not ((type == 'actor' and is_actor(ann_entity.iid)) or type == 'object' and is_object(ann_entity.iid)):
          errors.append(f'[{type}] wrong iid format {ann_entity.iid}'.encode('unicode_escape').decode('utf-8'))

        # check bbox
        if ann_entity.bbox.x < 0 or ann_entity.bbox.y < 0 or ann_entity.bbox.width <= 0 or ann_entity.bbox.height <= 0:
          errors.append(f'[{type}] wrong bbox size {ann_entity.bbox}')

      iids += [ann_entity.iid for ann_entity in anns_entity]

    # check duplicate iids
    if len(set(iids)) != len(iids):
      errors.append(f'[actor/object] duplicate iids {iids}')

    """ binary description & unary description """
    for i, type in enumerate(['binary description', 'unary description']):
      assert self.cn2en[ann_hoi_raw['task_result']['annotations'][i+2]['label']] == type
      anns_description_raw = ann_hoi_raw['task_result']['annotations'][i+2]['slotsChildren']
      anns_description = [Description(ann_description_raw, self.cn2en)
                          for ann_description_raw in anns_description_raw]

      for ann_description in anns_description:
        # check type
        if ann_description.type != type:
          errors.append(f'[{type}] wrong type {ann_description.type}')

        # check cname
        taxonomy = self.taxonomy_binary if type == 'binary description' else self.taxonomy_unary
        if ann_description.cname not in [x[0] for x in taxonomy]:
          errors.append(f'[{type}] unseen cname {ann_description.cname}')

        # check iids_associated
        if type == 'binary description':
          if ann_description.iids_associated[0] != '(' or \
             ann_description.iids_associated[-1] != ')' or \
             len(ann_description.iids_associated[1:-1].split('),(')) != 2:
            errors.append(f'[{type}] wrong iids_associated format {ann_description.iids_associated}')
            continue

          iids_src = ann_description.iids_associated[1:-1].split('),(')[0].split(',')
          iids_trg = ann_description.iids_associated[1:-1].split('),(')[1].split(',')
          if not are_entities(iids_src+iids_trg):
            errors.append(f'[{type}] wrong iids_associated format {iids_src} -> {iids_trg}')
            continue

          if not set(iids_src+iids_trg).issubset(iids):
            errors.append(f'[{type}] unseen iids_associated {set(iids_src+iids_trg)} in {iids}')
            continue

          cnames_binary = [x[0] for x in self.taxonomy_binary]
          if ann_description.cname not in cnames_binary:  # unseen cname
            continue
          index = cnames_binary.index(ann_description.cname)
          type_src, type_trg = self.taxonomy_binary[index][1:]
          if (type_src == 'actor' and not are_actors(iids_src)) or \
             (type_src == 'object' and not are_objects(iids_src)) or \
             (type_src == 'actor/object' and not are_entities(iids_src)) or \
             (type_trg == 'actor' and not are_actors(iids_trg)) or \
             (type_trg == 'object' and not are_objects(iids_trg)) or \
             (type_trg == 'actor/object' and not are_entities(iids_trg)):
            errors.append(f'[{type}] wrong iids_associated {ann_description.iids_associated} '
                          f'for types {type_src} -> {type_trg}')

        elif type == 'unary description':
          iids_src = ann_description.iids_associated.split(',')
          if not are_actors(iids_src):
            errors.append(f'[{type}] wrong iids_associated format {ann_description.iids_associated}')
            continue

          if not set(iids_src).issubset(iids):
            errors.append(f'[{type}] unseen iids_associated {set(iids_src)} in {iids}')

          cnames_unary = [x[0] for x in self.taxonomy_unary]
          index = cnames_unary.index(ann_description.cname)
          type_src = self.taxonomy_unary[index][1]
          if (type_src == 'actor' and not are_actors(iids_src)) or \
             (type_src == 'object' and not are_objects(iids_src)) or \
             (type_src == 'actor/object' and not are_entities(iids_src)):
            errors.append(f'[{type}] wrong iids_associated {ann_description.iids_associated} for types {type_src}')

    return errors

  def inspect(self, verbose=True):
    errors = []
    for iid_sact, ann_sact_raw in self.anns_sact_raw.items():
      errors_sact = self.__inspect_ann_sact(ann_sact_raw)
      if verbose and len(errors_sact) > 0:
        msg = errors_sact[0] if len(errors_sact) == 1 else '; '.join(errors_sact)
        print(f'Video {iid_sact}; {msg}')

      errors_hoi = []
      for ann_hoi_raw in ann_sact_raw:
        iid_hoi = self.get_iid_hoi(ann_hoi_raw)
        errors_hoi += self.__inspect_ann_hoi(ann_hoi_raw)
        if verbose and len(errors_hoi) > 0:
          msg = errors_hoi[0] if len(errors_hoi) == 1 else '; '.join(errors_hoi)
          print(f'Video {iid_sact} Image {iid_hoi}; {msg}')

      # error-free sub-activities
      if len(errors_sact) == 0 and len(errors_hoi) == 0:
        self.iids_sact.append(iid_sact)
      else:
        errors += errors_sact+errors_hoi

    print('\n ---------- REPORT (Phase 2) ----------')
    print(f'Number of error-free sub-activity instances: {len(self.anns_sact_raw)} -> {len(self.iids_sact)}')
    print(f'Number of errors: {len(errors)}')
