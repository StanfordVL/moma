import itertools
import json
import math
import os

from momaapi.data import Entity, Description
from momaapi import utils


def check_ann_image(ann_image_raw):
  errors = []

  assert len(ann_image_raw['task_result']['annotations']) == 4

  """ actor & object """
  iids = []
  for i, type in enumerate(['actor', 'object']):
    assert cn2en[ann_image_raw['task_result']['annotations'][i]['label']] == type
    anns_entity_raw = ann_image_raw['task_result']['annotations'][i]['slotsChildren']
    anns_entity = [Entity(ann_entity_raw, cn2en) for ann_entity_raw in anns_entity_raw]

    for ann_entity in anns_entity:
      # check type
      if ann_entity.type != type:
        errors.append(f'[{type}] wrong type {ann_entity.type}')

      # check cname
      if ann_entity.cname not in taxonomy_entity[type]:
        errors.append(f'[{type}] unseen cname {ann_entity.cname}')

      # check iid
      if not utils.is_entity(ann_entity.iid):
        errors.append(f'[{type}] wrong iid {ann_entity.iid}'.encode('unicode_escape').decode('utf-8'))

      # check bbox
      if ann_entity.bbox.x < 0 or ann_entity.bbox.y < 0 or ann_entity.bbox.width <= 0 or ann_entity.bbox.height <= 0:
        errors.append(f'[{type}] wrong bbox {ann_entity.bbox}')

    iids += [ann_entity.iid for ann_entity in anns_entity]

  # check duplicate iids
  if len(set(iids)) != len(iids):
    errors.append(f'[actor/object] duplicate iids {iids}')

  """ binary description & unary description """
  for i, type in enumerate(['binary description', 'unary description']):
    assert cn2en[ann_image_raw['task_result']['annotations'][i+2]['label']] == type
    anns_description_raw = ann_image_raw['task_result']['annotations'][i+2]['slotsChildren']
    anns_description = [Description(ann_description_raw, cn2en) for ann_description_raw in anns_description_raw]
    
    for ann_description in anns_description:
      # check type
      if ann_description.type != type:
        errors.append(f'[{type}] wrong type {ann_description.type}')

      # check cname
      if ann_description.cname not in [x[0] for x in taxonomy_description[type]]:
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
        if not utils.are_entities(iids_src+iids_trg):
          errors.append(f'[{type}] wrong iids_associated format {iids_src} -> {iids_trg}')
          continue

        if not set(iids_src+iids_trg).issubset(iids):
          errors.append(f'[{type}] unseen iids_associated {set(iids_src+iids_trg)} in {iids}')
          continue

        cnames_binary = [x[0] for x in taxonomy_binary]
        if ann_description.cname not in cnames_binary:  # unseen cname
          continue
        index = cnames_binary.index(ann_description.cname)
        type_src, type_trg = taxonomy_binary[index][1:]
        if (type_src == 'actor' and not utils.are_actors(iids_src)) or \
           (type_src == 'object' and not utils.are_objects(iids_src)) or \
           (type_src == 'actor/object' and not utils.are_entities(iids_src)) or \
           (type_trg == 'actor' and not utils.are_actors(iids_trg)) or \
           (type_trg == 'object' and not utils.are_objects(iids_trg)) or \
           (type_trg == 'actor/object' and not utils.are_entities(iids_trg)):
          errors.append(f'[{type}] wrong iids_associated {ann_description.iids_associated} '
                        f'for types {type_src} -> {type_trg}')

      elif type == 'unary description':
        iids_src = ann_description.iids_associated.split(',')
        if not utils.are_actors(iids_src):
          errors.append(f'[{type}] wrong iids_associated format {ann_description.iids_associated}')
          continue

        if not set(iids_src).issubset(iids):
          errors.append(f'[{type}] unseen iids_associated {set(iids_src)} in {iids}')

  return errors


def check_ann_video(ann_video_raw):
  errors = []

  anns_video_actor, anns_video_object = [], []
  for ann_image_raw in ann_video_raw:
    # actor
    anns_image_actor_raw = ann_image_raw['task_result']['annotations'][0]['slotsChildren']
    anns_image_actor = [Entity(ann_actor_raw, cn2en) for ann_actor_raw in anns_image_actor_raw]
    anns_video_actor += anns_image_actor

    # object
    anns_image_object_raw = ann_image_raw['task_result']['annotations'][1]['slotsChildren']
    anns_image_object = [Entity(ann_object_raw, cn2en) for ann_object_raw in anns_image_object_raw]
    anns_video_object += anns_image_object

  iids_video_actor = utils.sort(set([ann_video_actor.iid for ann_video_actor in anns_video_actor]))
  iids_video_object = utils.sort(set([ann_video_object.iid for ann_video_object in anns_video_object]))
  anns_instances_actor = [list(v) for _, v in itertools.groupby(anns_video_actor, lambda x: x.iid)]
  anns_instances_object = [list(v) for _, v in itertools.groupby(anns_video_object, lambda x: x.iid)]

  if not utils.is_consecutive(iids_video_actor):
    errors.append(f'[actor instance] iids not consecutive {iids_video_actor}')

  if not utils.is_consecutive(iids_video_object):
    errors.append(f'[object instance] iids not consecutive {iids_video_object}')

  for anns_instance_actor in anns_instances_actor:
    cnames = [ann_instance_actor.cname for ann_instance_actor in anns_instance_actor]
    if len(set(cnames)) != 1:
      errors.append(f'[actor instance] cname {set(cnames)} not unique')

  for anns_instance_object in anns_instances_object:
    cnames = [ann_instance_object.cname for ann_instance_object in anns_instance_object]
    if len(set(cnames)) != 1:
      errors.append(f'[object instance] cname {set(cnames)} not unique')

  return errors


def main():
  anns_video_raw = [list(v) for _, v in itertools.groupby(anns_raw, lambda x: x['id'])]

  report = {}
  for ann_video_raw in anns_video_raw:
    record = ann_video_raw[0]['task']['task_params']['record']
    id_video_real = record['attachment'].split('_')[-1][:-4].split('/')[0]
    id_image_to_timestamp_real = record['metadata']['additionalInfo']['framesTimestamp']
    num_images_real = len(ann_video_raw)
    ids_image_real = sorted(id_image_to_timestamp_real.keys(), key=int)
    assert ids_image_real[0] == '1' and ids_image_real[-1] == str(len(ids_image_real))

    errors = check_ann_video(ann_video_raw)
    # if len(errors) > 0:
    #   print(f'{id_video_real}: {errors[0] if len(errors) == 1 else errors}')
    report[id_video_real] = {}
    report[id_video_real]['instance'] = len(errors)

    for i, ann_image_raw in enumerate(ann_video_raw):
      record = ann_image_raw['task']['task_params']['record']

      id_video, timestamp = record['attachment'].split('_')[-1][:-4].split('/')
      timestamp = float(timestamp)/1000000
      assert id_video == id_video_real

      id_image_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']
      id_image = None
      for key in id_image_to_timestamp:
        if math.isclose(timestamp, id_image_to_timestamp[key], abs_tol=1e-6):
          id_image = key
      assert id_image_to_timestamp == id_image_to_timestamp_real
      assert id_image is not None and id_image == str(i+1)

      num_images = len(id_image_to_timestamp)
      assert num_images == num_images_real

      id_video = record['metadata']['additionalInfo']['videoName'].split('_')[-1].split('/')[0]
      assert id_video == id_video_real

      errors = check_ann_image(ann_image_raw)
      # if len(errors) > 0:
      #   print(f'{id_video}/{id_image}: {errors[0] if len(errors) == 1 else errors}')
      report[id_video][id_image] = len(errors)


if __name__ == '__main__':
  dir_anns = '/home/alan/ssd/moma/anns'

  anns_raw = []
  with open(os.path.join(dir_anns, 'MOMA-videos-all.jsonl'), 'r') as fs:
    for f in fs:
      anns_raw.append(json.loads(f))

  with open(os.path.join(dir_anns, 'taxonomy/actor.json'), 'r') as f:
    taxonomy_actor = json.load(f)
    taxonomy_actor = sorted(itertools.chain(*[taxonomy_actor[key] for key in taxonomy_actor]))
  with open(os.path.join(dir_anns, 'taxonomy/object.json'), 'r') as f:
    taxonomy_object = json.load(f)
    taxonomy_object = sorted(itertools.chain(*[taxonomy_object[key] for key in taxonomy_object]))
  with open(os.path.join(dir_anns, 'taxonomy/intransitive_action.json'), 'r') as f:
    taxonomy_ia = json.load(f)
    taxonomy_ia = sorted(itertools.chain(*[taxonomy_ia[key] for key in taxonomy_ia]))
    taxonomy_ia = [tuple(x) for x in taxonomy_ia]
  with open(os.path.join(dir_anns, 'taxonomy/transitive_action.json'), 'r') as f:
    taxonomy_ta = json.load(f)
    taxonomy_ta = sorted(itertools.chain(*[taxonomy_ta[key] for key in taxonomy_ta]))
    taxonomy_ta = [tuple(x) for x in taxonomy_ta]
  with open(os.path.join(dir_anns, 'taxonomy/attribute.json'), 'r') as f:
    taxonomy_att = json.load(f)
    taxonomy_att = sorted(itertools.chain(*[taxonomy_att[key] for key in taxonomy_att]))
    taxonomy_att = [tuple(x) for x in taxonomy_att]
  with open(os.path.join(dir_anns, 'taxonomy/relationship.json'), 'r') as f:
    taxonomy_rel = json.load(f)
    taxonomy_rel = sorted(itertools.chain(*[taxonomy_rel[key] for key in taxonomy_rel]))
    taxonomy_rel = [tuple(x) for x in taxonomy_rel]
  with open(os.path.join(dir_anns, 'taxonomy/cn2en.json'), 'r') as f:
    cn2en = json.load(f)

  taxonomy_entity = {'actor': taxonomy_actor, 'object': taxonomy_object}
  taxonomy_binary = taxonomy_ta+taxonomy_rel
  taxonomy_unary = taxonomy_ia+taxonomy_att
  taxonomy_description = {'binary description': taxonomy_binary, 'unary description': taxonomy_unary}

  main()
