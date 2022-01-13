import itertools
import json
import os
from pprint import pprint


def is_actor(iid):
  return iid.isupper() and len(iid) == 1


def is_object(iid):
  return iid.isnumeric() and 1 <= int(iid) <= 20


def is_actor_or_object(iid):
  return is_actor(iid) or is_object(iid)


def are_actors(iids):
  return all([is_actor(iid) for iid in iids])


def are_objects(iids):
  return all([is_object(iid) for iid in iids])


def are_actors_or_objects(iids):
  return all([is_actor_or_object(iid) for iid in iids])


class BBox:
  def __init__(self, bbox_raw):
    x_tl = round(bbox_raw['topLeft']['x'])
    y_tl = round(bbox_raw['topLeft']['y'])
    x_tr = round(bbox_raw['topRight']['x'])
    y_tr = round(bbox_raw['topRight']['y'])
    x_bl = round(bbox_raw['bottomLeft']['x'])
    y_bl = round(bbox_raw['bottomLeft']['y'])
    x_br = round(bbox_raw['bottomRight']['x'])
    y_br = round(bbox_raw['bottomRight']['y'])

    assert x_tl == x_bl and x_tr == x_br and y_tl == y_tr and y_bl == y_br, \
        '[BBox] coordinates inconsistent {}'.format(bbox_raw)

    # assert x_tl < x_tr and y_tl < y_bl, '[BBox] negative size {}'.format(bbox_raw)
    # FIXME: fix negative size error
    self.x = min(x_tl, x_tr)
    self.y = min(y_tl, y_bl)
    self.width = abs(x_tr-x_tl)
    self.height = abs(y_bl-y_tl)


class Entity:
  def __init__(self, entity_raw):
    type = cn2en[entity_raw['slot']['label']]
    assert type == 'actor' or type == 'object', '[Entity] wrong entity type {}'.format(type)

    cname = entity_raw['children'][0]['input']['value']
    # FIXME: fix type error
    if cname == '服务员':
      type = 'actor'
    if cname == '花洒' or cname == '飞盘' or cname == '钢琴' or cname == '足球门' or cname == '车' or cname == '纸' or \
       cname == '剃刀' or cname == '梳子' or cname == '吹风机' or cname == '相机' or \
       cname == '无法识别但确实和正在进行的动作相关的物体':
      type = 'object'
    # FIXME: fix typos
    cname = cname.replace(',', '，').replace(' ', '').replace('·', '')\
                 .replace('蓝球', '篮球').replace('蓝球框', '篮球框').replace('篮球框', '篮球框')\
                 .replace('篮子', '篮球框').replace('篮筐', '篮球框')
    assert cname in cn2en.keys(), '[Entity] unseen class name {}'.format(cname)
    cname = cn2en[cname]
    # FIXME: fix firefighting
    taxonomy_object.append('firefighting')
    assert (type == 'actor' and cname in taxonomy_actor) or (type == 'object' and cname in taxonomy_object), \
        '[Entity] wrong entity type {} for {}'.format(type, cname)

    iid = entity_raw['children'][1]['input']['value']
    # FIXME: fix typos
    iid = iid.replace(' ', '').replace('\n', '').upper()
    assert isinstance(iid, str) and ((type == 'actor' and is_actor(iid)) or (type == 'object' and is_object(iid))), \
           '[Entity] wrong entity iid \'{}\' for {} ({})'\
           .format(iid, cname, type).encode('unicode_escape').decode('utf-8')

    self.type = type
    self.cname = cname
    self.iid = iid
    self.bbox = BBox(entity_raw['slot']['plane'])


class Description:
  def __init__(self, description_raw):
    type = cn2en[description_raw['slot']['label']]
    assert type == 'binary description' or type == 'unary description', \
        '[Description] wrong entity type {}'.format(type)

    cname = description_raw['children'][0]['input']['value']
    # FIXME: fix typos
    cname = cname.replace('(', '（').replace(')', '）').replace(',', '，').replace(' ', '')\
                 .replace('保持低头的姿势', '保持低头的姿势且身体没有移动')\
                 .replace('保持蹲着的姿势', '保持蹲着的姿势且身体没有移动')\
                 .replace('保持跪姿并且没有移动', '保持跪姿并且身体没有移动')
    assert cname in cn2en.keys(), '[Description] unseen class name {}'.format(cname)
    cname = cn2en[cname]

    iids = description_raw['children'][1]['input']['value']
    iids = iids.replace('（', '(').replace('）', ')').replace('，', ',')\
               .replace(' ', '').replace('\n', '').replace('.', ',').replace(')(', '),(')

    if type == 'binary description':
      assert iids[0] == '(' and iids[-1] == ')' and len(iids[1:-1].split('),(')) == 2, \
          '[Binary Description] wrong iids format {}'.format(iids)
      iids = iids[1:-1].split('),(')
      iids_src, iids_trg = iids
      iids_src, iids_trg = iids_src.split(','), iids_trg.split(',')
      assert are_actors_or_objects(iids_src+iids_trg), \
          '[Binary Description] wrong iids {} -> {} for {} ({})'.format(iids_src, iids_trg, cname, type)

    elif type == 'unary description':
      iids = iids.split(',')
      assert are_actors(iids), '[Unary Description] wrong iids {} for {} ({})'.format(iids, cname, type)
    
    self.iids = iids
    self.type = type
    self.cname = cname


def check_issues(ann_raw):
  assert len(ann_raw['task_result']['annotations']) == 4

  # actor and object
  assert cn2en[ann_raw['task_result']['annotations'][0]['label']] == 'actor'
  anns_actor_raw = ann_raw['task_result']['annotations'][0]['slotsChildren']
  anns_actor = [Entity(ann_actor_raw) for ann_actor_raw in anns_actor_raw]
  iids_actor = [ann_actor.iid for ann_actor in anns_actor]
  assert all([ann_actor.type == 'actor' for ann_actor in anns_actor])

  # object
  assert cn2en[ann_raw['task_result']['annotations'][1]['label']] == 'object'
  anns_object_raw = ann_raw['task_result']['annotations'][1]['slotsChildren']
  anns_object = [Entity(ann_object_raw) for ann_object_raw in anns_object_raw]
  iids_object = [ann_object.iid for ann_object in anns_object]
  assert all([ann_object.type == 'object' for ann_object in anns_object])

  # binary description: transitive action and relationship
  assert cn2en[ann_raw['task_result']['annotations'][2]['label']] == 'binary description'
  anns_binary_description = []
  anns_binary_description_raw = ann_raw['task_result']['annotations'][2]['slotsChildren']
  for ann_binary_description_raw in anns_binary_description_raw:
    ann_binary_description = Description(ann_binary_description_raw)
    assert ann_binary_description.type == 'binary description', ann_binary_description.type
    assert len(ann_binary_description.iids) == 2, ann_binary_description.iids
    iids = ann_binary_description.iids[0].split(',')+ann_binary_description.iids[1].split(',')
    assert set(iids).issubset(set(iids_actor+iids_object)), \
        'binary description unseen iids {} vs seen iids {}'.format(ann_binary_description.iids, iids_actor+iids_object)
    anns_binary_description.append(ann_binary_description)

  # unary description: intransitive action and attribute
  assert cn2en[ann_raw['task_result']['annotations'][3]['label']] == 'unary description'
  anns_unary_description = []
  anns_unary_description_raw = ann_raw['task_result']['annotations'][3]['slotsChildren']
  for ann_unary_description_raw in anns_unary_description_raw:
    ann_unary_description = Description(ann_unary_description_raw)
    assert ann_unary_description.type == 'unary description', ann_unary_description.type
    assert set(ann_unary_description.iids).issubset(set(iids_actor)), \
        'unary description unseen iids {} vs seen iids {}'.format(ann_unary_description.iids, iids_actor)
    anns_unary_description.append(ann_unary_description)


def main():
  for i, ann_raw in enumerate(anns_raw):
    if i > 100:
      break

    try:
      check_issues(ann_raw)
    except AssertionError as msg:
      if str(msg).startswith('[Entity]'):
        continue
      print('{}: {}'.format(ann_raw['task']['task_params']['record']['metadata']['additionalInfo']['videoName'], msg))


if __name__ == '__main__':
  dir_moma = '/home/alan/ssd/moma'

  global anns_raw
  anns_raw = []
  with open(os.path.join(dir_moma, 'anns/MOMA-videos-all.jsonl'), 'r') as fs:
    for f in fs:
      anns_raw.append(json.loads(f))

  global taxonomy_actor, taxonomy_object, taxonomy_ia, taxonomy_ta, taxonomy_att, taxonomy_rel, cn2en
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

  main()
