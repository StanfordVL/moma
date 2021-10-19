import json
import os
from pprint import pprint


class bidict(dict):
  """
  A many-to-one bidirectional dictionary
  Reference: https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table
  """
  def __init__(self, *args, **kwargs):
    super(bidict, self).__init__(*args, **kwargs)
    self.inverse = {}
    for key, value in self.items():
      self.inverse.setdefault(value, set()).add(key)

  def __setitem__(self, key, value):
    if key in self:
      self.inverse[self[key]].remove(key)
    super(bidict, self).__setitem__(key, value)
    self.inverse.setdefault(value, set()).add(key)

  def __delitem__(self, key):
    self.inverse[self[key]].remove(key)
    if len(self.inverse[self[key]]) == 0:
      del self.inverse[self[key]]
    super(bidict, self).__delitem__(key)


def fix_cname(cname):
  return cname.strip().lower().replace('/ ', '/').replace('\u2019', "'")


def fix_iid(iid):
  return iid if isinstance(iid, str) else str(iid)


def fix_unary_rel(rel_raw):
  rel_raw = rel_raw.strip()
  if rel_raw[0] != '(' and rel_raw[-1] != ')':  # unary
    rel_raw = '('+rel_raw+')'
  return rel_raw


def fix_binary_rel(rel_raw):
  if rel_raw == '(C,(2)':
    rel_raw += ')'
  return rel_raw


def main(parse_video=True, parse_graph=True):
  """ load data """
  # moma_dir = '/vision/u/zelunluo/moma'
  moma_dir = '../'
  video_anns_fname = 'video_anns_phase1_processed.json'
  graph_anns_fname = 'graph_anns.json'

  with open(os.path.join(moma_dir, video_anns_fname), 'r') as f:
    video_anns = json.load(f)
  with open(os.path.join(moma_dir, graph_anns_fname), 'r') as f:
    graph_anns = json.load(f)

  """ preprocess data """
  # fix typos
  for act_iid, act_ann in video_anns.items():
    for i in range(len(act_ann['subactivity'])):
      video_anns[act_iid]['subactivity'][i]['class'] = fix_cname(video_anns[act_iid]['subactivity'][i]['class'])
      video_anns[act_iid]['subactivity'][i]['subactivity_instance_id'] = fix_iid(video_anns[act_iid]['subactivity'][i]['subactivity_instance_id'])

  # fix typos & list -> dict
  graph_anns_new = {}
  for graph_ann in graph_anns:
    for i in range(len(graph_ann['atomic_actions'])):
      graph_ann['atomic_actions'][i]['actor_id'] = fix_unary_rel(graph_ann['atomic_actions'][i]['actor_id'])
    for i in range(len(graph_ann['relationships'])):
      graph_ann['relationships'][i]['description'] = fix_binary_rel(graph_ann['relationships'][i]['description'])
    graph_anns_new.setdefault(graph_ann['trim_video_id'], []).append(graph_ann)
  graph_anns = graph_anns_new


  """ init """
  cnames = {}
  cnames2 = {}
  iids = {}
  for act_iid, act_ann in video_anns.items():
    act_cname = act_ann['class']
    for sact_ann in act_ann['subactivity']:
      cnames.setdefault(act_cname, set()).add(sact_ann['class'])
      sact_iid = sact_ann['subactivity_instance_id']
      iids.setdefault(act_iid, set()).add(sact_iid)

  for ag_iid, ag_ann in graph_anns.items():
    for sg_ann in ag_ann:
      for actor in sg_ann['actors']:
        cnames2.setdefault('actor', set()).add(actor['class'])
      for object in sg_ann['objects']:
        cnames2.setdefault('object', set()).add(object['class'])
      for rel in sg_ann['atomic_actions']:
        cnames2.setdefault('rel', set()).add(rel['class'])
      for rel in sg_ann['relationships']:
        cnames2.setdefault('rel', set()).add(rel['class'])

  """ parse video annotations """
  if parse_video:
    # activity classes
    act_cnames = set()
    for act_iid, act_ann in video_anns.items():
      act_cnames.add(act_ann['class'])
    print(act_cnames, len(act_cnames), '\n')

    # sub-activity classes
    sact_cnames = set()
    for act_iid, act_ann in video_anns.items():
      for sact_ann in act_ann['subactivity']:
        sact_cnames.add(sact_ann['class'])
    print(sact_cnames, len(sact_cnames), '\n')

    # check for redundancy in annotations
    for act_iid, act_ann in video_anns.items():
      # make sure video_id is redundant
      assert act_iid == act_ann['video_id']

      # make sure fps is redundant
      for sact_ann in act_ann['subactivity']:
        assert sact_ann['fps'] == act_ann['fps']

    # make sure sub-activity classes from different activity classes are mutually exclusive
    act_cnames = list(cnames.keys())
    for i in range(len(act_cnames)):
      for j in range(i+1, len(act_cnames)):
        sact_cnames1 = cnames[act_cnames[i]]
        sact_cnames2 = cnames[act_cnames[j]]
        assert len(set(sact_cnames1).intersection(set(sact_cnames2))) == 0

    # make sure activity instance id is unique
    act_iids = []
    for act_iid, act_ann in video_anns.items():
      act_iids.append(act_iid)
    assert len(act_iids) == len(set(act_iids))
    print("#activity instances = {}".format(len(act_iids)))

    # make sure sub-activity instance id is unique
    sact_iids = []
    for act_iid, act_ann in video_anns.items():
      for sact_ann in act_ann['subactivity']:
        sact_iid = sact_ann['subactivity_instance_id']
        sact_iids.append(sact_iid)
    assert len(sact_iids) == len(set(sact_iids))
    print("#sub-activity instances = {}".format(len(sact_iids)))

  """ parse graph annotations """
  if parse_graph:
    # ag_key == sact_iid
    # dict_keys(['actors', 'atomic_actions', 'relationships', 'objects', 'raw_video_id', 'trim_video_id', 'graph_id', 'frame_dim', 'subactivity', 'frame_timestamp', 'activity'])

    # make sure scene graph key (instance id) is unique
    sg_keys = []
    for ag_key, ag_ann in graph_anns.items():
      for sg_ann in ag_ann:
        sg_key = sg_ann['graph_id']
        sg_keys.append(sg_key)
    assert len(sg_keys) == len(set(sg_keys))
    print("#scene graph instances = {}\n".format(len(sg_keys)))

    # check for redundancy in annotations
    for ag_key, ag_ann in graph_anns.items():
      # make sure frame_dim is redundant
      frame_dim = ag_ann[0]['frame_dim']
      for sg_ann in ag_ann:
        assert sg_ann['frame_dim'] == frame_dim

    # check relationship formats
    for ag_key, ag_ann in graph_anns.items():
      for sg_ann in ag_ann:
        for aact in sg_ann['atomic_actions']:
          rel_raw = aact['actor_id']
          assert 2*rel_raw.count(',')+3 == len(rel_raw), '{}: {} vs. {}'.format(rel_raw, len(rel_raw), 2*rel_raw.count(',')+1)
        for aact in sg_ann['relationships']:
          rel_raw = aact['description']
          assert rel_raw[0] == '(' and rel_raw[-1] == ')' and rel_raw.count('(') == 2 and rel_raw.count(')') == 2, rel_raw

    # check entity instance ids
    for ag_key, ag_ann in graph_anns.items():
      actor_iids = set()
      object_iids = set()
      unary_rels = set()
      binary_rels = set()

      for sg_ann in ag_ann:
        for actor in sg_ann['actors']:
          actor_iids.add(actor['id_in_video'])
        for object in sg_ann['objects']:
          object_iids.add(object['id_in_video'])
        for rel in sg_ann['atomic_actions']:
          unary_rels.add(rel['actor_id'])
        for rel in sg_ann['relationships']:
          binary_rels.add(rel['description'])

      entity_iids = set.union(actor_iids, object_iids)
      rel_iids = set.union(*[set(rel) for rel in unary_rels.union(binary_rels)])-{'(', ')', ','}

      # error 1: entity iids do not match those in relationships
      # assert iids == rel_iids, ag_key
      if entity_iids != rel_iids:
        # print(sorted(actor_iids))
        # print(sorted(object_iids))
        # print(sorted(iids))
        # print(sorted(rel_iids))
        print('entity iids do not match those in relationships: {}'.format(ag_key))

      # error 2: iids not consecutive
      if len(actor_iids) > 0 and (sorted(actor_iids)[0] != 'A' or sorted(actor_iids)[-1] != chr(ord('A')+len(actor_iids)-1)):
        print('actor iids not consecutive: {}'.format(ag_key))
      if len(object_iids) > 0 and (sorted(object_iids)[0] != '1' or sorted(object_iids)[-1] != str(len(object_iids))):
        print('object iids not consecutive: {}'.format(ag_key))

  # save as json: cnames mapping
  for act_cname in cnames.keys():
    cnames[act_cname] = sorted(cnames[act_cname])  # set -> list
  with open(os.path.join(moma_dir, './cnames.json'), 'w') as f:
    json.dump(cnames, f, indent=4, sort_keys=True, ensure_ascii=False)

  for key in cnames2.keys():
    cnames2[key] = sorted(cnames2[key])  # set -> list
  with open(os.path.join(moma_dir, './cnames2.json'), 'w') as f:
    json.dump(cnames2, f, indent=4, sort_keys=True, ensure_ascii=False)

  # save as json: instance id mapping
  for act_iid in sorted(iids.keys()):
    iids[act_iid] = sorted(iids[act_iid])  # set -> list
  with open(os.path.join(moma_dir, './iids.json'), 'w') as f:
    json.dump(iids, f, indent=4, sort_keys=True)


if __name__ == '__main__':
  main()
