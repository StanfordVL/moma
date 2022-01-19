import itertools
import json
from multiprocessing import cpu_count, Pool
import os

from momaapi import utils


def fix_issues():
  # remove activity instances without sub-activities
  for iid_act in list(anns_act):
    ann_sact = anns_act[iid_act]['subactivity']
    if len(ann_sact) == 0:
      del anns_act[iid_act]

  # fix activity fps
  anns_act['ifAZ3iwtjik']['fps'] = 24
  for ann_sact in anns_act['ifAZ3iwtjik']['subactivity']:
    ann_sact['fps'] = 24

  # fix incorrect activity boundary format
  anns_act['Kuhc6od_huU']['crop_end'] = '00:01:18'

  # fix incorrect activity boundary
  anns_act['0HxGaLh6YM4']['crop_end'] = '00:10:00'
  anns_act['3SU6a9jrGgo']['crop_start'] = '00:00:36'
  anns_act['3SU6a9jrGgo']['crop_end'] = '00:09:02'
  anns_act['4pptxtS9K7E']['crop_end'] = '00:08:22'
  anns_act['K50aHl3UcU0']['crop_end'] = '00:05:38'
  anns_act['g3PRJgSfFuk']['crop_end'] = '00:10:00'
  anns_act['pA6FaBIa3iM']['crop_end'] = '00:02:30'
  anns_act['y6GNrpcXtqM']['crop_end'] = '00:09:53'

  # fix incorrect sub-activity boundary format
  anns_act['Kuhc6od_huU']['subactivity'][1]['end'] = '00:01:00'    # 6390

  # fix incorrect sub-activity boundary
  anns_act['pA6FaBIa3iM']['subactivity'][-2]['end'] = '00:02:09'   # 8443
  anns_act['4pptxtS9K7E']['subactivity'][2]['end'] = '00:01:33'    # 734
  anns_act['y6GNrpcXtqM']['subactivity'][3]['end'] = '00:02:24'    # 2747
  anns_act['y6GNrpcXtqM']['subactivity'][4]['start'] = '00:02:30'  # 2747
  anns_act['y6GNrpcXtqM']['subactivity'][4]['end'] = '00:02:37'    # 2747
  anns_act['y6GNrpcXtqM']['subactivity'][5]['start'] = '00:02:52'  # 2747
  anns_act['y6GNrpcXtqM']['subactivity'][5]['end'] = '00:02:58'    # 2747
  anns_act['K50aHl3UcU0']['subactivity'][-2]['end'] = '00:05:16'   # 12929

  """ corrections below affect phase 2 """

  return anns_act


def check_ann_act(iid_act, ann_act):
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
  # metadata_video = ffmpeg.probe(file_video)['streams'][0]
  # fps_video = round(Fraction(metadata_video['avg_frame_rate']))
  # assert ann_act['fps'] == fps_video, 'activity fps inconsistent'
  # assert all([ann_sact['fps'] == fps_video for ann_sact in anns_sact]), 'sub-activity fps inconsistent'

  # make sure the temporal boundary is in the right format
  assert utils.is_hms(ann_act['crop_start']) and utils.is_hms(ann_act['crop_end']), \
      f"incorrect activity boundary format {ann_act['crop_start']}, {ann_act['crop_end']}"

  # make sure the activity temporal boundary is within the video and the length is positive
  start_act = utils.hms2s(ann_act['crop_start'])  # inclusive
  end_act = utils.hms2s(ann_act['crop_end'])  # exclusive
  # end_video = math.ceil(float(metadata_video['duration']))
  # assert 0 <= start_act < end_act <= end_video, \
  #     f'activity boundary exceeds video boundary: 0 <= {start_act} < {end_act} <= {end_video}'

  errors = []
  iids_sact_bad = []
  start_sact_last, end_sact_last = start_act, start_act
  anns_sact = sorted(anns_sact, key=lambda x: utils.hms2s(x['start']))
  for ann_sact in anns_sact:
    iid_sact = ann_sact['subactivity_instance_id']

    # make sure the temporal boundary is in the right format
    assert utils.is_hms(ann_sact['start']) and utils.is_hms(ann_sact['end']), \
        f"incorrect sub-activity boundary format {ann_sact['start']}, {ann_sact['end']}"

    # make sure the sub-activity temporal boundary is after the previous one
    start_sact = utils.hms2s(ann_sact['start'])
    end_sact = utils.hms2s(ann_sact['end'])
    if end_sact_last > start_sact:
      if end_sact_last >= end_sact:
        errors.append(f"[{iid_act}; {iid_sact}] completely overlapped sub-activity boundaries "
                      f"({utils.s2hms(start_sact_last)}, {utils.s2hms(end_sact_last)}) and "
                      f"({utils.s2hms(start_sact)}, {utils.s2hms(end_sact)})")
      else:
        errors.append(f"[{iid_act}; {iid_sact}] partially overlapped sub-activity boundaries "
                      f"({utils.s2hms(start_sact_last)}, {utils.s2hms(end_sact_last)}) and "
                      f"({utils.s2hms(start_sact)}, {utils.s2hms(end_sact)})")
      iids_sact_bad.append(iid_sact)
    start_sact_last = start_sact
    end_sact_last = end_sact

    # make sure the sub-activity temporal boundary is within the activity and the length is positive
    if not (start_act <= start_sact < end_sact <= end_act):
      errors.append(f'[{iid_act}; {iid_sact}] '
                    f'incorrect sub-activity boundary {utils.s2hms(start_act)} <= {utils.s2hms(start_sact)} < '
                    f'{utils.s2hms(end_sact)} <= {utils.s2hms(end_act)}')
      iids_sact_bad.append(iid_sact)

  iids_act_bad = None if len(errors) == 0 else iid_act
  return errors, iids_act_bad, iids_sact_bad


def check_errors():
  # check video files
  fnames_video_all = os.listdir(os.path.join(dir_moma, dname_video))
  assert all([fname_video.endswith('.mp4') for fname_video in fnames_video_all])

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

  # check each ann_act
  p = Pool(processes=cpu_count()-1)
  errors, iids_act_bad, iids_sact_bad = zip(*p.starmap(check_ann_act, anns_act.items()))
  errors = list(itertools.chain(*[error for error in errors if len(error) > 0]))
  iids_act_bad = [iid_act_bad for iid_act_bad in iids_act_bad if iid_act_bad is not None]
  iids_sact_bad = list(itertools.chain(*[iid_sact_bad for iid_sact_bad in iids_sact_bad if len(iid_sact_bad) > 0]))

  # compare with phase 2
  iids_sact_phase1 = [ann_sact['subactivity_instance_id'] for iid_act, ann_act in anns_act.items()
                                                          for ann_sact in ann_act['subactivity']]
  # print(set(iids_sact_phase1)-set(iids_sact_phase2))
  # print(set(iids_sact_phase2)-set(iids_sact_phase1))
  # print(set(iids_sact_phase2)-(set(iids_sact_phase1)-set(iids_sact_bad)))
  print(f'number of missing sub-activity annotations = '
        f'{len(set(iids_sact_phase2)-(set(iids_sact_phase1)-set(iids_sact_bad)))}')
  print(f'number of sub-activity annotations with errors = {len(errors)}')
  # pprint(errors)

  # return errors that are relevant to phase 2
  dict_errors = {int(error.split(']')[0].split('; ')[1]):error for error in errors}
  iids_sact_missing = list(set(iids_sact_phase2)-(set(iids_sact_phase1)-set(iids_sact_bad)))
  errors = [dict_errors[iid_sact_missing] for iid_sact_missing in iids_sact_missing]
  return errors


def parse_anns():
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
  fix_issues()
  errors = check_errors()
  # dict_cnames, dict_iids = parse_anns()

  # save as json
  with open(os.path.join(dir_moma, './errors.json'), 'w') as f:
    json.dump(errors, f, indent=4, sort_keys=True, ensure_ascii=False)
  # with open(os.path.join(dir_moma, './dict_cnames.json'), 'w') as f:
  #   json.dump(dict_cnames, f, indent=4, sort_keys=True, ensure_ascii=False)
  # with open(os.path.join(dir_moma, './dict_iids.json'), 'w') as f:
  #   json.dump(dict_iids, f, indent=4, sort_keys=True)
  # with open(os.path.join(dir_moma, './iids_act_bad.json'), 'w') as f:
  #   json.dump(iids_act_bad, f, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
  dir_moma = '/home/alan/ssd/moma'
  dname_video = 'videos_cn'
  dname_ann = 'anns'
  fname_ann = 'video_anns_phase1_processed.json'
  fname_iids_sact_phase2 = 'iids_sact_phase2.json'

  with open(os.path.join(dir_moma, dname_ann, fname_ann), 'r') as f:
    anns_act = json.load(f)

  with open(os.path.join(dir_moma, dname_ann, fname_iids_sact_phase2), 'r') as f:
    iids_sact_phase2 = json.load(f)
    iids_sact_phase2 = [int(x) for x in iids_sact_phase2]

  main()
