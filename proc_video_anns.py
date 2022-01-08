import datetime
import ffmpeg
from fractions import Fraction
import json
import os
from pprint import pprint


def to_seconds(hhmmss):
  dt = datetime.datetime.strptime(hhmmss, '%H:%M:%S')
  seconds = datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second).total_seconds()
  return seconds


def fix_issues(anns_act):
  # remove activity instances without sub-activities
  for key_act in list(anns_act):
    ann_sact = anns_act[key_act]['subactivity']
    if len(ann_sact) == 0:
      del anns_act[key_act]
  print('Removed activity instances without sub-activities, {} activity instances remaining'.format(len(anns_act)))

  # fix known issues
  anns_act['ifAZ3iwtjik']['fps'] = 24
  for ann_sact in anns_act['ifAZ3iwtjik']['subactivity']:
    ann_sact['fps'] = 24

  anns_act['Kuhc6od_huU']['crop_end'] = '00:01:00'
  anns_act['Kuhc6od_huU']['subactivity'][1]['end'] = '00:10:00'

  return anns_act


def check_issues(anns_act):
  # check video files
  fnames_video_all = os.listdir(os.path.join(dir_moma, dname_video))
  assert all([fname_video.endswith('.mp4') for fname_video in fnames_video_all])

  keys_sact = []
  for key_act, ann_act in anns_act.items():
    # make sure key_acts are consistent
    assert key_act == ann_act['video_id'], key_act

    # make sure the corresponding video exist
    anns_sact = ann_act['subactivity']
    fname_video = anns_sact[0]['orig_vid']
    file_video = os.path.join(dir_moma, dname_video, fname_video)
    assert os.path.isfile(file_video)

    # check fps
    metadata_ffmpeg = ffmpeg.probe(file_video)['streams'][0]
    fps_ffmpeg = round(Fraction(metadata_ffmpeg['avg_frame_rate']))
    assert ann_act['fps'] == fps_ffmpeg, key_act
    assert all([ann_sact['fps'] == fps_ffmpeg for ann_sact in anns_sact]), key_act

    # check start (inclusive) and end (exclusive) times
    end_ffmpeg = float(metadata_ffmpeg['duration'])
    start_act = to_seconds(ann_act['crop_start'])
    end_act = to_seconds(ann_act['crop_end'])
    # assert start_act >= 0 and end_act <= end_ffmpeg, '{}: {}-{} vs. 0-{}'.format(key_act, start_act, end_act, end_ffmpeg)
    if not (start_act >= 0 and end_act <= end_ffmpeg):  # activity
      print('{}: {}-{} (activity) vs. 0-{} (ffmpeg)'.format(key_act, start_act, end_act, end_ffmpeg))
    end_sact_last = start_act
    for ann_sact in anns_sact:
      start_sact = to_seconds(ann_sact['start'])
      end_sact = to_seconds(ann_sact['end'])
      # assert start_act <= end_sact_last <= start_sact < end_sact <= end_act, key_act
      if not (start_act <= end_sact_last <= start_sact < end_sact <= end_act):  # sub-activity
        print('{}: {} <= {} <= {} < {} <= {}'.format(key_act, start_act, end_sact_last, start_sact, end_sact, end_act))
      end_sact_last = end_sact

      keys_sact.append(ann_sact['subactivity_instance_id'])

  # make sure key_sacts are unique integers
  assert len(keys_sact) == len(set(keys_sact))
  assert all([isinstance(key_sact, int) for key_sact in keys_sact])


def parse_anns(anns_act):
  cnames = {}
  keys = {}
  
  for key_act, ann_act in anns_act.items():
    cname_act = ann_act['class']
    for ann_sact in ann_act['subactivity']:
      cnames.setdefault(cname_act, set()).add(ann_sact['class'])
      key_sact = ann_sact['subactivity_instance_id']
      keys.setdefault(key_act, set()).add(key_sact)

  pprint(cnames)
  pprint(keys)

  


def main():
  with open(os.path.join(dir_moma, dname_ann, fname_ann)) as f:
    anns_act = json.load(f)
  print('{} activity instances'.format(len(anns_act)))

  anns_act = fix_issues(anns_act)
  # check_issues(anns_act)
  parse_anns(anns_act)


if __name__ == '__main__':
  dir_moma = '/home/alan/ssd/moma'
  dname_video = 'videos_cn'
  dname_ann = 'anns'
  fname_ann = 'video_anns_updated.json'
  # fname_ann = 'video_anns_phase1_processed.json'

  main()
