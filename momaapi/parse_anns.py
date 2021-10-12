from itertools import chain
import json
import os
from pprint import pprint

import data_structure


class bidict(dict):
  def __init__(self, mapping):
    super(bidict, self).__init__()

    self._dict = {}
    self.inverse = {}
    assert isinstance(mapping, dict)
    for key, value in mapping.items():
      self[key] = value

  def __setitem__(self, key, value):
    self._dict[key] = value
    self.inverse.setdefault(value, set()).add(key)

  def __getitem__(self, key):
    return self._dict[key]

  def __delitem__(self, key):
    self.inverse[self._dict[key]].remove(key)
    del self._dict[key]

  def __repr__(self):
    return repr(self._dict)


def fix_cname(cname):
  return cname.strip().lower().replace('/ ', '/').replace('\u2019', "'")


def load_cnames(moma_dir, fname='cnames.json'):
  with open(os.path.join(moma_dir, fname), 'r') as f:
    cnames = json.load(f)

  act_cnames = sorted(cnames.keys())
  sact_cnames = sorted(set(chain(*cnames.values())))
  act_cid2cnames = bidict({i:act_cname for i, act_cname in enumerate(act_cnames)})
  sact_cid2cnames = bidict({i:sact_cname for i, sact_cname in enumerate(sact_cnames)})
  aact2act_cnames = {}
  for aact_cname in cnames:
    for sact_cname in cnames[aact_cname]:
      if sact_cname in aact2act_cnames:
        assert aact2act_cnames[sact_cname] == aact_cname
      else:
        aact2act_cnames[sact_cname] = aact_cname

  pprint(aact2act_cnames)
  aact2act_cnames = bidict(aact2act_cnames)

  return cnames, act_cid2cnames, sact_cid2cnames, aact2act_cnames


def main(parse_video=False, parse_graph=True):
  # moma_dir = '/vision/u/zelunluo/moma'
  moma_dir = '../'
  video_anns_fname = 'video_anns_phase1_processed.json'
  graph_anns_fname = 'graph_anns.json'

  with open(os.path.join(moma_dir, video_anns_fname), 'r') as f:
    video_anns = json.load(f)
  with open(os.path.join(moma_dir, graph_anns_fname), 'r') as f:
    graph_anns = json.load(f)

  # modify anns
  for act_key, act_ann in video_anns:
    for i in range(len(act_ann['subactivity'])):
      video_anns[act_key]['subactivity'][i]['class'] = fix_cname(video_anns[act_key]['subactivity'][i]['class'])

  graph_anns_new = {}
  for graph_ann in graph_anns:
    graph_anns_new[graph_ann['trim_video_id']] = graph_ann
  graph_anns = graph_anns_new

  if parse_video:
    for act_key, act_ann in video_anns.items():
      # make sure video_id is redundant
      assert act_key == act_ann['video_id']

      # make sure fps is redundant
      for sact_ann in act_ann['subactivity']:
        assert sact_ann['fps'] == act_ann['fps']

    # activity classes
    act_cnames = set()
    for key, act_ann in video_anns.items():
      act_cnames.add(act_ann['class'])
    print(act_cnames, len(act_cnames), '\n')

    # sub-activity classes
    sact_cnames = set()
    for act_key, act_ann in video_anns.items():
      for sact_ann in act_ann['subactivity']:
        sact_cnames.add(sact_ann['class'])
    print(sact_cnames, len(sact_cnames), '\n')

    # make sure sub-activity classes from different activity classes are mutually exclusive
    cnames = {}
    for act_key, act_ann in video_anns.items():
      act_cname = act_ann['class']
      if act_cname not in cnames:
        cnames[act_cname] = set()
      for sact_ann in act_ann['subactivity']:
        cnames[act_cname].add(sact_ann['class'])
    # pprint(cnames)

    act_cnames = list(cnames.keys())
    for i in range(len(act_cnames)):
      for j in range(i+1, len(act_cnames)):
        sact_cnames1 = cnames[act_cnames[i]]
        sact_cnames2 = cnames[act_cnames[j]]
        assert len(set(sact_cnames1).intersection(set(sact_cnames2))) == 0

    # make sure activity key is unique
    act_keys = []
    for act_key, act_ann in video_anns.items():
      act_keys.append(act_key)
    assert len(act_keys) == len(set(act_keys))
    print("#activity instances = {}".format(len(act_keys)))

    # make sure sub-activity key is unique
    sact_keys = []
    for act_key, act_ann in video_anns.items():
      for sact_ann in act_ann['subactivity']:
        sact_key = sact_ann['subactivity_instance_id']
        sact_keys.append(sact_key)
    assert len(sact_keys) == len(set(sact_keys))
    print("#sub-activity instances = {}".format(len(sact_keys)))

    # save as json
    for act_cname in cnames.keys():
      sact_cnames = list(cnames[act_cname])
      cnames[act_cname] = sact_cnames
    with open(os.path.join(moma_dir, '../cnames.json'), 'w') as f:
      json.dump(cnames, f, indent=4, sort_keys=True)

  act_ann = data_structure.ActAnn(list(video_anns.values())[0])

  if parse_graph:
    print(len(graph_anns))
    print(sorted(graph_anns[0].keys()))
    pprint(graph_anns[200])

    # make sure graph id is unique
    graph_ids = []
    for key, graph_ann in graph_anns.items():
      graph_ids.append(graph_ann['graph_id'])
    assert len(graph_ids) == len(set(graph_ids))
    print("#graphs = {}".format(len(graph_ids)))

    # make sure frame timestamp starts at zero and has the same length as the sub-activity videos
    cur_video_id = graph_anns[0]['trim_video_id']
    cur_frame_id = 0
    for graph_ann in graph_anns:
      if graph_ann['trim_video_id'] != cur_video_id:  # new sub-activity
        cur_video_id = graph_ann['trim_video_id']
        assert graph_ann['frame_timestamp'] == 0
        cur_frame_id = 0

      assert graph_ann['frame_timestamp'] == cur_frame_id, '{} vs. {}'.format(graph_ann['frame_timestamp'], cur_frame_id)
      cur_frame_id += 1


if __name__ == '__main__':
  main()
