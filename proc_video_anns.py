import datetime
import ffmpeg
from fractions import Fraction
import json
import math
from multiprocessing import cpu_count, Pool
import os
from pprint import pprint


def hms2s(hms):
  dt = datetime.datetime.strptime(hms, '%H:%M:%S')
  seconds = datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second).total_seconds()
  return seconds


def is_hms(hms):
  try:
    datetime.datetime.strptime(hms, '%H:%M:%S')
    return True
  except ValueError:
    return False


def fix_issues(anns_act):
  # remove activity instances without sub-activities
  for iid_act in list(anns_act):
    ann_sact = anns_act[iid_act]['subactivity']
    if len(ann_sact) == 0:
      del anns_act[iid_act]

  # fix activity fps
  anns_act['ifAZ3iwtjik']['fps'] = 24
  for ann_sact in anns_act['ifAZ3iwtjik']['subactivity']:
    ann_sact['fps'] = 24

  # fix activity boundary format
  anns_act['Kuhc6od_huU']['crop_end'] = '00:01:00'
  anns_act['Kuhc6od_huU']['subactivity'][1]['end'] = '00:10:00'

  return anns_act


def check_ann_act(iid_act, ann_act):
  try:
    # make sure iid_acts is consistent
    assert iid_act == ann_act['video_id'], 'iid_acts inconsistent'

    # make sure there is at least one sub-activity
    anns_sact = ann_act['subactivity']
    assert len(anns_sact) > 0, 'no sub-activity'

    # make sure the corresponding video exist
    fname_video = anns_sact[0]['orig_vid']
    file_video = os.path.join(dir_moma, dname_video, fname_video)
    assert os.path.isfile(file_video), 'video file does not exit'

    # make sure fps is consistent
    metadata_video = ffmpeg.probe(file_video)['streams'][0]
    fps_video = round(Fraction(metadata_video['avg_frame_rate']))
    assert ann_act['fps'] == fps_video, 'activity fps inconsistent'
    assert all([ann_sact['fps'] == fps_video for ann_sact in anns_sact]), 'sub-activity fps inconsistent'

    # make sure the activity temporal boundary is in the right format
    assert is_hms(ann_act['crop_start']) and is_hms(ann_act['crop_end']), 'incorrect activity boundary format'

    # make sure the activity temporal boundary is within the video and the length is positive
    start_act = hms2s(ann_act['crop_start'])  # inclusive
    end_act = hms2s(ann_act['crop_end'])  # exclusive
    end_video = math.ceil(float(metadata_video['duration']))
    assert 0 <= start_act < end_act <= end_video, \
        'incorrect activity boundary: 0 <= {} < {} <= {}'.format(start_act, end_act, end_video)

    end_sact_last = start_act
    for i, ann_sact in enumerate(anns_sact):
      # make sure the activity temporal boundary is in the right format
      assert is_hms(ann_sact['start']) and is_hms(ann_sact['end']), 'incorrect sub-activity boundary format'

      # make sure the sub-activity temporal boundary is after the previous one
      start_sact = hms2s(ann_sact['start'])
      end_sact = hms2s(ann_sact['end'])
      assert end_sact_last <= start_sact, \
          'incorrect sub-activity boundary {} <= {}'.format(end_sact_last, start_sact)
      end_sact_last = end_sact

      # make sure the sub-activity temporal boundary is within the activity and the length is positive
      assert start_act <= start_sact < end_sact <= end_act, \
          'incorrect sub-activity boundary {} <= {} < {} <= {}'.format(start_act, start_sact, end_sact, end_act)

      # make sure iid_sact is consecutive
      assert ann_sact['subactivity_instance_id'] == anns_sact[0]['subactivity_instance_id']+i, \
          'iid_sact not consecutive'

  except AssertionError as msg:
    print('{}: {}'.format(iid_act, msg))
    return iid_act


def check_issues(anns_act):
  # check video files
  fnames_video_all = os.listdir(os.path.join(dir_moma, dname_video))
  assert all([fname_video.endswith('.mp4') for fname_video in fnames_video_all])

  p = Pool(processes=cpu_count()-1)
  iids_act_bad = p.starmap(check_ann_act, anns_act.items())
  iids_act_bad = [iid_act_bad for iid_act_bad in iids_act_bad if iid_act_bad is not None]
  print('{} bad videos'.format(len(iids_act_bad)))
  for iid_act_bad in iids_act_bad:
    anns_act.pop(iid_act_bad)

  # make sure iid_sacts are unique integers
  iids_sact = [ann_sact['subactivity_instance_id'] for iid_act in anns_act
                                                   for ann_sact in anns_act[iid_act]['subactivity']]
  assert len(iids_sact) == len(set(iids_sact))

  # make sure sub-activity classes from different activity classes are mutually exclusive
  dict_cnames = {}
  for iid_act, ann_act in anns_act.items():
    cname_act = ann_act['class']
    for ann_sact in ann_act['subactivity']:
      dict_cnames.setdefault(cname_act, set()).add(ann_sact['class'])
  cnames_act = list(dict_cnames.keys())
  for i in range(len(cnames_act)):
    for j in range(i+1, len(cnames_act)):
      cnames_sact_1 = dict_cnames[cnames_act[i]]
      cnames_sact_2 = dict_cnames[cnames_act[j]]
      assert len(cnames_sact_1.intersection(cnames_sact_2)) == 0

  return anns_act, iids_act_bad


def parse_anns(anns_act):
  dict_cnames = {}
  dict_iids = {}
  for iid_act, ann_act in anns_act.items():
    cname_act = ann_act['class']
    for ann_sact in ann_act['subactivity']:
      dict_cnames.setdefault(cname_act, set()).add(ann_sact['class'])
      iid_sact = ann_sact['subactivity_instance_id']
      dict_iids.setdefault(iid_act, set()).add(iid_sact)

  # set -> list
  for cname_act in dict_cnames.keys():
    dict_cnames[cname_act] = sorted(dict_cnames[cname_act])
  for iid_act in sorted(dict_iids.keys()):
    dict_iids[iid_act] = sorted(dict_iids[iid_act])

  # pprint(dict_cnames, width=150)
  # pprint(dict_iids)
    
  return dict_cnames, dict_iids


def main():
  with open(os.path.join(dir_moma, dname_ann, fname_ann), 'r') as f:
    anns_act = json.load(f)

  anns_act = fix_issues(anns_act)
  anns_act, iids_act_bad = check_issues(anns_act)
  dict_cnames, dict_iids = parse_anns(anns_act)

  # save as json
  with open(os.path.join(dir_moma, './dict_cnames.json'), 'w') as f:
    json.dump(dict_cnames, f, indent=4, sort_keys=True, ensure_ascii=False)
  with open(os.path.join(dir_moma, './dict_iids.json'), 'w') as f:
    json.dump(dict_iids, f, indent=4, sort_keys=True)
  with open(os.path.join(dir_moma, './iids_act_bad.json'), 'w') as f:
    json.dump(iids_act_bad, f, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
  dir_moma = '/home/alan/ssd/moma'
  dname_video = 'videos_cn'
  dname_ann = 'anns'
  fname_ann = 'video_anns_updated.json'
  # fname_ann = 'video_anns_phase1_processed.json'

  main()
