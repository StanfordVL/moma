import datetime
import json
import os
from pprint import pprint

import parse_anns


""" Terminology:
 - activity: act
 - sub-activity: sact
 - atomic action: aact
"""


def hms2s(t):
  h, m, s = [int(i) for i in t.split(':')]
  return int(datetime.timedelta(hours=h, minutes=m, seconds=s).total_seconds())


class ActAnn:
  """
   - key: a key which uniquely identifies an activity annotation
   - cid: activity class id
   - start: start frame
   - end: end frame
   - fname: file name
   - fps: frames per second
   - sact_anns: sub-activity annotations
  """
  def __init__(self, raw_ann):
    self.key, self.cid, self.start, self.end, self.fname, self.sact_anns = None, None, None, None, None, None

    self.key = raw_ann['video_id']
    self.cid = act_cid2cnames.inverse[parse_anns.fix_cname(raw_ann['class'])]
    self.start = hms2s(raw_ann['crop_start'])*raw_ann['fps']
    self.end = hms2s(raw_ann['crop_end'])*raw_ann['fps']
    self.fname = raw_ann['subactivity'][0]['orig_vid']
    self.fps = raw_ann['fps']
    self.sact_anns = [SactAnn(x, self) for x in raw_ann['subactivity']]


class SactAnn:
  """
   - key: a key which uniquely identifies a sub-activity annotation
   - cid: sub-activity class id
   - start: start frame
   - end: end frame
   - aact_ann: action hypergraph annotation
   - act_ann: reference to its parent activity annotation
  """
  def __init__(self, raw_ann, act_ann):
    self.key, self.cid, self.fname, self.act_ann = None, None, None, None

    self.key = raw_ann['subactivity_instance_id']
    self.cid = sact_cid2cnames.inverse[parse_anns.fix_cname(raw_ann['class'])]
    self.start = hms2s(raw_ann['start'])*raw_ann['fps']
    self.end = hms2s(raw_ann['end'])*raw_ann['fps']
    # self.aact_ann =
    self.act_ann = act_ann


# class AGAnn:


def main():
  moma_dir = '../'
  act_cid2cnames, sact_cid2cnames, aact2act_cnames = parse_anns.load_cnames(moma_dir)[1:]
  cnames, act_cid2cnames, sact_cid2cnames = parse_anns.load_cnames(moma_dir)
  print(act_cid2cnames)
  print(sact_cid2cnames)
  print(aact2act_cnames)

  # graph_anns_fname = 'graph_anns.json'
  # video_anns_fname = 'video_anns_phase1_processed.json'
  #
  # with open(os.path.join(moma_dir, graph_anns_fname), 'r') as f:
  #   graph_anns = json.load(f)
  #
  # with open(os.path.join(moma_dir, video_anns_fname), 'r') as f:
  #   video_anns = json.load(f)
  #
  # act_ann = ActAnn(list(video_anns.values())[0])
  # print('\n\n')
  # pprint(vars(act_ann))
  # print('\n\n')
  # pprint(vars(act_ann.sact_anns[0]))
  # print('\n\n')
  # print(act_ann == act_ann.sact_anns[0].act_ann, act_ann is act_ann.sact_anns[0].act_ann)


if __name__ == '__main__':
  main()
