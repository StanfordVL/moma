from collections import defaultdict
import ffmpeg
from fractions import Fraction
import json
import math
import os

from .data import *
from .utils import *


class AnnPhase1:
  def __init__(self, dir_moma, fname_ann):
    with open(os.path.join(dir_moma, 'anns', fname_ann), 'r') as f:
      anns_act = json.load(f)

    with open(os.path.join(dir_moma, 'anns/taxonomy/act_sact.json'), 'r') as f:
      taxonomy = json.load(f)
    with open(os.path.join(dir_moma, 'anns/taxonomy/cn2en.json'), 'r') as f:
      cn2en = json.load(f)

    self.anns_act = anns_act  # dict
    self.taxonomy = taxonomy
    self.cn2en = cn2en
    self.dir_moma = dir_moma
    self.id_act_to_ids_sact = {}
    self.metadata = {}

    self.__fix()

  def __fix(self):
    # remove activity instances without sub-activities
    for id_act in list(self.anns_act):
      ann_sact = self.anns_act[id_act]['subactivity']
      if len(ann_sact) == 0:
        del self.anns_act[id_act]

    # fix activity fps
    self.anns_act['ifAZ3iwtjik']['fps'] = 24
    for i in range(len(self.anns_act['ifAZ3iwtjik']['subactivity'])):
      self.anns_act['ifAZ3iwtjik']['subactivity'][i]['fps'] = 24

    # fix incorrect activity boundary format
    self.anns_act['Kuhc6od_huU']['crop_end'] = '00:01:18'

    # fix incorrect activity boundary
    self.anns_act['0HxGaLh6YM4']['crop_end'] = '00:10:00'
    self.anns_act['3SU6a9jrGgo']['crop_start'] = '00:00:36'
    self.anns_act['3SU6a9jrGgo']['crop_end'] = '00:09:02'
    self.anns_act['4pptxtS9K7E']['crop_end'] = '00:08:22'
    self.anns_act['K50aHl3UcU0']['crop_end'] = '00:05:38'
    self.anns_act['g3PRJgSfFuk']['crop_end'] = '00:10:00'
    self.anns_act['pA6FaBIa3iM']['crop_end'] = '00:02:30'
    self.anns_act['y6GNrpcXtqM']['crop_end'] = '00:09:53'

    lookup = {}
    for id_act, ann_act in self.anns_act.items():
      for i, ann_sact in enumerate(ann_act['subactivity']):
        id_sact = self.get_id_sact(ann_sact)
        lookup[id_sact] = (id_act, i)

    # fix incorrect sub-activity boundary format
    self.anns_act[lookup['6390'][0]]['subactivity'][lookup['6390'][1]]['end'] = '00:01:00'

    # fix incorrect sub-activity boundary
    self.anns_act[lookup['8443'][0]]['subactivity'][lookup['8443'][1]]['end'] = '00:02:09'
    self.anns_act[lookup['734'][0]]['subactivity'][lookup['734'][1]]['end'] = '00:01:33'
    self.anns_act[lookup['2747'][0]]['subactivity'][lookup['2747'][1]]['end'] = '00:02:24'
    self.anns_act[lookup['2748'][0]]['subactivity'][lookup['2748'][1]]['start'] = '00:02:30'
    self.anns_act[lookup['2748'][0]]['subactivity'][lookup['2748'][1]]['end'] = '00:02:37'
    self.anns_act[lookup['2749'][0]]['subactivity'][lookup['2749'][1]]['start'] = '00:02:52'
    self.anns_act[lookup['2749'][0]]['subactivity'][lookup['2749'][1]]['end'] = '00:02:58'
    self.anns_act[lookup['12929'][0]]['subactivity'][lookup['12929'][1]]['end'] = '00:05:16'

    """ corrections below affect phase 2 """
    # fix incorrect sub-activity boundary
    self.anns_act[lookup['3361'][0]]['subactivity'][lookup['3361'][1]]['start'] = '00:05:33'
    self.anns_act[lookup['5730'][0]]['subactivity'][lookup['5730'][1]]['start'] = '00:03:56'
    self.anns_act[lookup['6239'][0]]['subactivity'][lookup['6239'][1]]['end'] = '00:03:49'
    self.anns_act[lookup['6679'][0]]['subactivity'][lookup['6679'][1]]['start'] = '00:01:05'
    self.anns_act[lookup['9534'][0]]['subactivity'][lookup['9534'][1]]['end'] = '00:00:09'
    self.anns_act[lookup['11065'][0]]['subactivity'][lookup['11065'][1]]['end'] = '00:04:20'

    # remove overlapping sub-activity
    ids_sact_rm = ['27', '198', '199', '653', '1535', '1536', '3775', '4024', '5531', '5629',
                    '5729', '6178', '6478', '7073', '7074', '7076', '7350', '9713', '10926', '10927',
                    '11168', '11570', '12696', '12697', '15225', '15403', '15579', '15616']
    for id_sact_rm in sorted(ids_sact_rm, key=int, reverse=True):  # remove in descending index order
      del self.anns_act[lookup[id_sact_rm][0]]['subactivity'][lookup[id_sact_rm][1]]

  @staticmethod
  def get_id_act(ann_act):
    return ann_act['video_id']

  @staticmethod
  def get_id_sact(ann_sact):
    return str(ann_sact['subactivity_instance_id'])

  @staticmethod
  def get_cname_act(ann_act):
    return ann_act['class']

  def get_cname_sact(self, ann_sact):
    return self.cn2en[ann_sact['filename'].split('_')[0]]

  def __inspect_anns_act(self):
    # check video files
    fnames_video_all = os.listdir(os.path.join(self.dir_moma, 'videos/raw_all'))
    assert all([fname_video.endswith('.mp4') for fname_video in fnames_video_all])

    # make sure ids_sact are unique integers across different activities
    ids_sact = [self.get_id_sact(ann_sact) for id_act in self.anns_act
                 for ann_sact in self.anns_act[id_act]['subactivity']]
    assert len(ids_sact) == len(set(ids_sact))

    # make sure sub-activity classes from different activity classes are mutually exclusive
    dict_cnames = {}
    for id_act, ann_act in self.anns_act.items():
      cname_act = ann_act['class']
      for ann_sact in ann_act['subactivity']:
        dict_cnames.setdefault(cname_act, set()).add(ann_sact['class'])
    cnames_act = list(dict_cnames.keys())
    for i in range(len(cnames_act)):
      for j in range(i+1, len(cnames_act)):
        cnames_sact_1 = dict_cnames[cnames_act[i]]
        cnames_sact_2 = dict_cnames[cnames_act[j]]
        assert len(cnames_sact_1.intersection(cnames_sact_2)) == 0

  def __inspect_ann_act(self, id_act, ann_act):
    # make sure the class name exists
    cname_act = self.get_cname_act(ann_act)
    assert cname_act in self.taxonomy.keys(), f"unseen class name {cname_act}"

    # make sure id_act is consistent
    assert id_act == self.get_id_act(ann_act), 'inconsistent id_act'

    # make sure there is at least one sub-activity
    anns_sact = ann_act['subactivity']
    assert len(anns_sact) > 0, 'no sub-activity'

    # make sure the corresponding video exist
    fname_video = anns_sact[0]['orig_vid']
    file_video = os.path.join(self.dir_moma, f'videos/raw_all/{fname_video}')
    assert os.path.isfile(file_video), 'video file does not exit'

    # make sure fps is consistent
    probe = ffmpeg.probe(file_video)
    self.metadata[id_act] = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    fps_video = round(Fraction(self.metadata[id_act]['avg_frame_rate']))
    assert ann_act['fps'] == fps_video, 'inconsistent activity fps'
    assert all([ann_sact['fps'] == fps_video for ann_sact in anns_sact]), 'inconsistent sub-activity fps'

    # make sure the temporal boundary is in the right format
    assert is_hms(ann_act['crop_start']) and is_hms(ann_act['crop_end']), \
        f"incorrect activity boundary format {ann_act['crop_start']}, {ann_act['crop_end']}"

    # make sure the activity temporal boundary is within the video and the length is positive
    start_act = hms2s(ann_act['crop_start'])  # inclusive
    end_act = hms2s(ann_act['crop_end'])  # exclusive
    end_video = math.ceil(float(self.metadata[id_act]['duration']))
    assert 0 <= start_act < end_act <= end_video, \
        f'activity boundary exceeds video boundary: 0 <= {start_act} < {end_act} <= {end_video}'

    errors = defaultdict(list)
    start_sact_last, end_sact_last = start_act, start_act
    anns_sact = sorted(anns_sact, key=lambda x: hms2s(x['start']))
    for ann_sact in anns_sact:
      # make sure the class name exists
      cname_sact = self.get_cname_sact(ann_sact)
      assert cname_sact in self.taxonomy[cname_act], \
          f'unseen class name {cname_sact} in {cname_act}'

      id_sact = self.get_id_sact(ann_sact)

      # make sure the temporal boundary is in the right format
      assert is_hms(ann_sact['start']) and is_hms(ann_sact['end']), \
          f"incorrect sub-activity boundary format {ann_sact['start']}, {ann_sact['end']}"

      # make sure the sub-activity temporal boundary is after the previous one
      start_sact = hms2s(ann_sact['start'])
      end_sact = hms2s(ann_sact['end'])
      if end_sact_last > start_sact:
        if end_sact_last >= end_sact:
          errors[id_sact].append(f'completely overlapped sub-activity boundaries '
                                  f'({s2hms(start_sact_last)}, {s2hms(end_sact_last)}) and '
                                  f'({s2hms(start_sact)}, {s2hms(end_sact)})')
        else:
          errors[id_sact].append(f'partially overlapped sub-activity boundaries '
                                  f'({s2hms(start_sact_last)}, {s2hms(end_sact_last)}) and '
                                  f'({s2hms(start_sact)}, {s2hms(end_sact)})')
      start_sact_last = start_sact
      end_sact_last = end_sact

      # make sure the sub-activity temporal boundary is within the activity and the length is positive
      if not (start_act <= start_sact < end_sact <= end_act):
        errors[id_sact].append(f'incorrect sub-activity boundary '
                                f'{s2hms(start_act)} <= {s2hms(start_sact)} < '
                                f'{s2hms(end_sact)} <= {s2hms(end_act)}')

    errors = defaultdict_to_dict(errors)
    return errors

  def inspect(self, verbose=True):
    self.__inspect_anns_act()
    for id_act, ann_act in self.anns_act.items():
      ids_sact = [self.get_id_sact(ann_sact) for ann_sact in ann_act['subactivity']]
      errors = self.__inspect_ann_act(id_act, ann_act)

      if verbose:
        for id_sact, msg in errors.items():
          print(f'Activity {id_act} Sub-activity {id_sact}; {msg[0] if len(msg) == 1 else msg}')

      # error-free activities and sub-activities
      ids_sact = [id_sact for id_sact in ids_sact if id_sact not in errors.keys()]
      self.id_act_to_ids_sact[id_act] = ids_sact

    num_acts_before = len(self.anns_act)
    num_sacts_before = sum([len(ann_act['subactivity']) for ann_act in self.anns_act.values()])
    num_acts_after = len(self.id_act_to_ids_sact)
    num_sacts_after = sum([len(ids_sact) for ids_sact in self.id_act_to_ids_sact.values()])

    print('\n ---------- REPORT (Phase 1) ----------')
    print(f'Number of error-free activity instances: {num_acts_before} -> {num_acts_after}')
    print(f'Number of error-free sub-activity instances: {num_sacts_before} -> {num_sacts_after}')

  def get_distribution(self):
    distribution = defaultdict(lambda: defaultdict(int))

    for ann_act in self.anns_act.values():
      cname_act = self.get_cname_act(ann_act)
      for ann_sact in ann_act['subactivity']:
        cname_sact = self.get_cname_sact(ann_sact)
        distribution[cname_act][cname_sact] += 1

    distribution = defaultdict_to_dict(distribution)
    return distribution
