import json
import os
from pprint import pprint


class Entity:
  """
   - x, y: the upper-left coordinates of the bounding box
   - width, height: the dimensions of the bounding box

  """
  def __init__(self, ann_entity):
    x_tl = round(ann_entity['bbox']['topLeft']['x'])
    y_tl = round(ann_entity['bbox']['topLeft']['y'])
    x_tr = round(ann_entity['bbox']['topRight']['x'])
    y_tr = round(ann_entity['bbox']['topRight']['y'])
    x_bl = round(ann_entity['bbox']['bottomLeft']['x'])
    y_bl = round(ann_entity['bbox']['bottomLeft']['y'])
    x_br = round(ann_entity['bbox']['bottomRight']['x'])
    y_br = round(ann_entity['bbox']['bottomRight']['y'])
    assert x_tl == x_bl and x_tr == x_br and y_tl == y_tr and y_bl == y_br, \
        'entity bbox inconsistent {}'.format(ann_entity['bbox'])
    assert x_tl < x_tr and y_tl < y_bl, 'entity bbox negative size {}'.format(ann_entity['bbox'])
    self.x = x_tl
    self.y = y_tl
    self.width = x_tr-x_tl
    self.height = y_bl-y_tl

    self.type = ann_entity['label_type']
    assert self.type == 'actor' or self.type == 'object', 'wrong entity type'

    self.iid = ann_entity['id_in_video']
    assert isinstance(self.iid, str) and \
           ((self.type == 'actor' and self.iid.isupper() and len(self.iid) == 1) or
            (self.type == 'object' and self.iid.isnumeric() and len(self.iid) <= 2)), \
           'wrong entity iid \'{}\' for type {}'.format(self.iid, self.type).encode('unicode_escape').decode('utf-8')


def main():
  with open(os.path.join(dir_moma, dname_ann, fname_ann), 'r') as f:
    anns_int = json.load(f)

  with open(os.path.join(dir_moma, 'dict_cnames.json')) as f:
    dict_cnames_video_anns = json.load(f)
  with open(os.path.join(dir_moma, 'dict_iids.json')) as f:
    dict_iids_video_anns = json.load(f)
  with open(os.path.join(dir_moma, 'iids_act_bad.json')) as f:
    iids_act_bad_video = json.load(f)

  dict_cnames, dict_iids = {}, {}
  for ann_int in anns_int:
    dict_cnames.setdefault(ann_int['activity'], set()).add(ann_int['subactivity'])
    dict_iids.setdefault(ann_int['raw_video_id'], set()).add(int(ann_int['trim_video_id']))
  for cname_act in sorted(dict_cnames.keys()):
    dict_cnames[cname_act] = sorted(dict_cnames[cname_act])
  for iid_act in sorted(dict_iids.keys()):
    dict_iids[iid_act] = sorted(dict_iids[iid_act])
  for iid_act_bad_video in iids_act_bad_video:
    dict_iids.pop(iid_act_bad_video)

  # check cnames_act and cnames_sact are consistent with video_anns
  assert dict_cnames.keys() == dict_cnames_video_anns.keys()  # cnames_act
  # todo: check cnames_sact

  # check iids_act and iids_sact are consistent with video_anns
  assert dict_iids.keys() == dict_iids_video_anns.keys()  # iids_act
  for iid in dict_iids:
    assert len(dict_iids[iid]) <= len(dict_iids_video_anns[iid])
    try:
      assert dict_iids[iid] == dict_iids_video_anns[iid], \
          '{}: {} sub-activities in graph_anns vs. {} sub-activities in video_anns'\
          .format(iid, len(dict_iids[iid]), len(dict_iids_video_anns[iid]))
    except AssertionError as msg:
      # print(msg)
      pass

  """ check per-frame errors """
  for ann_int in anns_int:
    # check actor and object
    try:
      for ann_entity in ann_int['actors']+ann_int['objects']:
        entity = Entity(ann_entity)
    except AssertionError as msg:
      print('{} {}: {}'.format(ann_int['raw_video_id'], ann_int['trim_video_id'], msg))

    # check state and atomic action
    iids_actor = [ann_actor['id_in_video'] for ann_actor in ann_int['actor']]
    iids_object = [ann_actor['id_in_video'] for ann_actor in ann_int['actor']]
    # try:
    #   for ann_entity in ann_int['atomic_actions']+ann_int['relationships']:
    #     entity = Entity(ann_entity)
    # except AssertionError as msg:
    #   print('{} {}: {}'.format(ann_int['raw_video_id'], ann_int['trim_video_id'], msg))

  """ check per-sub-activity errors """


if __name__ == '__main__':
  dir_moma = '/home/alan/ssd/moma'
  dname_video = 'videos_cn'
  dname_ann = 'anns'
  fname_ann = 'graph_anns_1228.json'

  main()
