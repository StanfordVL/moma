from collections import defaultdict
import ffmpeg
from fractions import Fraction
import itertools
import json
import math
import os
from pprint import pprint

import moma


class AnnPhase1:
  def __init__(self, dir_moma, fname_ann):
    with open(os.path.join(dir_moma, 'anns', fname_ann), 'r') as f:
      anns_act = json.load(f)

    with open(os.path.join(dir_moma, 'anns/taxonomy/act_sact.json'), 'r') as f:
      taxonomy = json.load(f)
    with open(os.path.join(dir_moma, 'anns/taxonomy/cn2en.json'), 'r') as f:
      cn2en = json.load(f)

    self.anns_act = anns_act
    self.taxonomy = taxonomy
    self.cn2en = cn2en
    self.dir_moma = dir_moma
    self.iid_act_to_iids_sact = {}
    self.num_acts_raw = len(anns_act)
    self.num_sacts_raw = sum([len(ann_act['subactivity']) for ann_act in anns_act.values()])

    self.__fix()

  def __fix(self):
    # remove activity instances without sub-activities
    for iid_act in list(self.anns_act):
      ann_sact = self.anns_act[iid_act]['subactivity']
      if len(ann_sact) == 0:
        del self.anns_act[iid_act]

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
    for iid_act, ann_act in self.anns_act.items():
      for i, ann_sact in enumerate(ann_act['subactivity']):
        iid_sact = self.get_iid_sact(ann_sact)
        lookup[iid_sact] = (iid_act, i)

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
    iids_sact_rm = ['27', '198', '199', '653', '1535', '1536', '3775', '4024', '5531', '5629',
                    '5729', '6178', '6478', '7073', '7074', '7076', '7350', '9713', '10926', '10927',
                    '11168', '11570', '12696', '12697', '15225', '15403', '15579', '15616']
    for iid_sact_rm in sorted(iids_sact_rm, key=int, reverse=True):  # remove in descending index order
      del self.anns_act[lookup[iid_sact_rm][0]]['subactivity'][lookup[iid_sact_rm][1]]

  @staticmethod
  def get_iid_act(ann_act):
    return ann_act['video_id']

  @staticmethod
  def get_iid_sact(ann_sact):
    return str(ann_sact['subactivity_instance_id'])

  @staticmethod
  def get_cname_act(ann_act):
    return ann_act['class']

  def get_cname_sact(self, ann_sact):
    return self.cn2en[ann_sact['filename'].split('_')[0]]

  def __inspect_anns_act(self):
    # check video files
    fnames_video_all = os.listdir(os.path.join(self.dir_moma, 'videos'))
    assert all([fname_video.endswith('.mp4') for fname_video in fnames_video_all])

    # make sure iids_sact are unique integers across different activities
    iids_sact = [self.get_iid_sact(ann_sact) for iid_act in self.anns_act
                 for ann_sact in self.anns_act[iid_act]['subactivity']]
    assert len(iids_sact) == len(set(iids_sact))

    # make sure sub-activity classes from different activity classes are mutually exclusive
    dict_cnames = {}
    for iid_act, ann_act in self.anns_act.items():
      cname_act = ann_act['class']
      for ann_sact in ann_act['subactivity']:
        dict_cnames.setdefault(cname_act, set()).add(ann_sact['class'])
    cnames_act = list(dict_cnames.keys())
    for i in range(len(cnames_act)):
      for j in range(i+1, len(cnames_act)):
        cnames_sact_1 = dict_cnames[cnames_act[i]]
        cnames_sact_2 = dict_cnames[cnames_act[j]]
        assert len(cnames_sact_1.intersection(cnames_sact_2)) == 0

  def __inspect_ann_act(self, iid_act, ann_act):
    # make sure the class name exists
    cname_act = self.get_cname_act(ann_act)
    assert cname_act in self.taxonomy.keys(), f"unseen class name {cname_act}"

    # make sure iid_act is consistent
    assert iid_act == self.get_iid_act(ann_act), 'inconsistent iid_act'

    # make sure there is at least one sub-activity
    anns_sact = ann_act['subactivity']
    assert len(anns_sact) > 0, 'no sub-activity'

    # make sure the corresponding video exist
    fname_video = anns_sact[0]['orig_vid']
    file_video = os.path.join(self.dir_moma, 'videos', fname_video)
    assert os.path.isfile(file_video), 'video file does not exit'

    # make sure fps is consistent
    metadata_video = ffmpeg.probe(file_video)['streams'][0]
    fps_video = round(Fraction(metadata_video['avg_frame_rate']))
    assert ann_act['fps'] == fps_video, 'inconsistent activity fps'
    assert all([ann_sact['fps'] == fps_video for ann_sact in anns_sact]), 'inconsistent sub-activity fps'

    # make sure the temporal boundary is in the right format
    assert moma.is_hms(ann_act['crop_start']) and moma.is_hms(ann_act['crop_end']), \
        f"incorrect activity boundary format {ann_act['crop_start']}, {ann_act['crop_end']}"

    # make sure the activity temporal boundary is within the video and the length is positive
    start_act = moma.hms2s(ann_act['crop_start'])  # inclusive
    end_act = moma.hms2s(ann_act['crop_end'])  # exclusive
    end_video = math.ceil(float(metadata_video['duration']))
    assert 0 <= start_act < end_act <= end_video, \
        f'activity boundary exceeds video boundary: 0 <= {start_act} < {end_act} <= {end_video}'

    errors = defaultdict(list)
    start_sact_last, end_sact_last = start_act, start_act
    anns_sact = sorted(anns_sact, key=lambda x: moma.hms2s(x['start']))
    for ann_sact in anns_sact:
      # make sure the class name exists
      cname_sact = self.get_cname_sact(ann_sact)
      assert cname_sact in self.taxonomy[cname_act], \
          f'unseen class name {cname_sact} in {cname_act}'

      iid_sact = self.get_iid_sact(ann_sact)

      # make sure the temporal boundary is in the right format
      assert moma.is_hms(ann_sact['start']) and moma.is_hms(ann_sact['end']), \
          f"incorrect sub-activity boundary format {ann_sact['start']}, {ann_sact['end']}"

      # make sure the sub-activity temporal boundary is after the previous one
      start_sact = moma.hms2s(ann_sact['start'])
      end_sact = moma.hms2s(ann_sact['end'])
      if end_sact_last > start_sact:
        if end_sact_last >= end_sact:
          errors[iid_sact].append(f'completely overlapped sub-activity boundaries '
                                  f'({moma.s2hms(start_sact_last)}, {moma.s2hms(end_sact_last)}) and '
                                  f'({moma.s2hms(start_sact)}, {moma.s2hms(end_sact)})')
        else:
          errors[iid_sact].append(f'partially overlapped sub-activity boundaries '
                                  f'({moma.s2hms(start_sact_last)}, {moma.s2hms(end_sact_last)}) and '
                                  f'({moma.s2hms(start_sact)}, {moma.s2hms(end_sact)})')
      start_sact_last = start_sact
      end_sact_last = end_sact

      # make sure the sub-activity temporal boundary is within the activity and the length is positive
      if not (start_act <= start_sact < end_sact <= end_act):
        errors[iid_sact].append(f'incorrect sub-activity boundary '
                                f'{moma.s2hms(start_act)} <= {moma.s2hms(start_sact)} < '
                                f'{moma.s2hms(end_sact)} <= {moma.s2hms(end_act)}')

    errors = moma.defaultdict_to_dict(errors)
    return errors

  def inspect(self, verbose=True):
    self.__inspect_anns_act()
    for iid_act, ann_act in self.anns_act.items():
      iids_sact = [self.get_iid_sact(ann_sact) for ann_sact in ann_act['subactivity']]
      errors = self.__inspect_ann_act(iid_act, ann_act)

      if verbose:
        for iid_sact, msg in errors.items():
          print(f'Activity {iid_act} Sub-activity {iid_sact}; {msg[0] if len(msg) == 1 else msg}')

      # error-free activities and sub-activities
      iids_sact = [iid_sact for iid_sact in iids_sact if iid_sact not in errors.keys()]
      self.iid_act_to_iids_sact[iid_act] = iids_sact

    num_acts = len(self.iid_act_to_iids_sact)
    num_sacts = sum([len(iids_sact) for iids_sact in self.iid_act_to_iids_sact.values()])

    print('\n ---------- REPORT (Phase 1) ----------')
    print(f'Number of activity instances: {self.num_acts_raw} -> {num_acts}')
    print(f'Number of sub-activity instances: {self.num_sacts_raw} -> {num_sacts}')

  def get_distribution(self):
    distribution = defaultdict(lambda: defaultdict(int))

    for ann_act in self.anns_act.values():
      cname_act = self.get_cname_act(ann_act)
      for ann_sact in ann_act['subactivity']:
        cname_sact = self.get_cname_sact(ann_sact)
        distribution[cname_act][cname_sact] += 1

    distribution = moma.defaultdict_to_dict(distribution)
    return distribution


class AnnPhase2:
  def __init__(self, dir_moma, fname_ann):
    anns_raw = []
    with open(os.path.join(dir_moma, 'anns', fname_ann), 'r') as fs:
      for f in fs:
        anns_raw.append(json.loads(f))
    anns_video_raw = [list(v) for _, v in itertools.groupby(anns_raw, lambda x: x['id'])]

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

    taxonomy_unary = taxonomy_ia+taxonomy_att
    taxonomy_binary = taxonomy_ta+taxonomy_rel

    self.anns_video_raw = anns_video_raw
    self.taxonomy_actor = taxonomy_actor
    self.taxonomy_object = taxonomy_object
    self.taxonomy_ia = taxonomy_ia
    self.taxonomy_ta = taxonomy_ta
    self.taxonomy_att = taxonomy_att
    self.taxonomy_rel = taxonomy_rel
    self.taxonomy_unary = taxonomy_unary
    self.taxonomy_binary = taxonomy_binary
    self.cn2en = cn2en
    self.iids_sact = []

    self.__fix()

  @staticmethod
  def __trim(ann_video_raw, num_images_trim, is_start):
    num_images = len(ann_video_raw)
    record = ann_video_raw[0]['task']['task_params']['record']
    duration = record['metadata']['additionalInfo']['duration']-num_images_trim
    id_image_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']

    for j in range(num_images_trim):
      id_image_to_timestamp.pop(str(num_images-j))

    for j in range(num_images):
      ann_video_raw[j]['task']['task_params']['record']['metadata']['additionalInfo']['duration'] = duration
      ann_video_raw[j]['task']['task_params']['record']['metadata']['additionalInfo'][
        'framesTimestamp'] = id_image_to_timestamp

    if is_start:
      ann_video_raw = ann_video_raw[num_images_trim:]
    else:
      for j in range(num_images-num_images_trim):
        ann_video_raw[num_images_trim+j]['task']['task_params']['record']['attachment'] = \
          ann_video_raw[j]['task']['task_params']['record']['attachment']
      ann_video_raw = ann_video_raw[:-num_images_trim]

    return ann_video_raw

  def __fix(self):
    iids_sact_fix = ['3361', '5730', '6239', '6679', '9534', '11065']
    lookup = {}
    for i, ann_video_raw in enumerate(self.anns_video_raw):
      iid_sact = self.get_id_video(ann_video_raw)
      if iid_sact in iids_sact_fix:
        lookup[iid_sact] = i

    self.anns_video_raw[lookup['3361']] = self.__trim(self.anns_video_raw[lookup['3361']], 2, True)    # start: 31 -> 33
    self.anns_video_raw[lookup['5730']] = self.__trim(self.anns_video_raw[lookup['5730']], 1, True)    # start: 55 -> 56
    self.anns_video_raw[lookup['6239']] = self.__trim(self.anns_video_raw[lookup['6239']], 10, False)    # end: 59 -> 49
    self.anns_video_raw[lookup['6679']] = self.__trim(self.anns_video_raw[lookup['6679']], 5, True)    # start: 00 -> 05
    self.anns_video_raw[lookup['9534']] = self.__trim(self.anns_video_raw[lookup['9534']], 10, False)    # end: 19 -> 09
    self.anns_video_raw[lookup['11065']] = self.__trim(self.anns_video_raw[lookup['11065']], 10, False)  # end: 30 -> 20

    iids_sact_rm = ['27', '198', '199', '653', '1535', '1536', '4024', '5729', '6178', '6478',
                    '7074', '7076', '7350', '9713', '11570', '12697', '15225', '15403', '15579', '15616']
    self.anns_video_raw = [ann_video_raw for ann_video_raw in self.anns_video_raw
                           if self.get_id_video(ann_video_raw) not in iids_sact_rm]

  @staticmethod
  def get_id_video(ann_video_raw):
    record = ann_video_raw[0]['task']['task_params']['record']
    id_video = record['attachment'].split('_')[-1][:-4].split('/')[0]
    return id_video

  @staticmethod
  def get_id_image(ann_image_raw):
    record = ann_image_raw['task']['task_params']['record']
    id_video, timestamp = record['attachment'].split('_')[-1][:-4].split('/')
    timestamp = float(timestamp)/1000000
    id_image_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']
    id_image = None
    for key in id_image_to_timestamp:
      if math.isclose(timestamp, id_image_to_timestamp[key], abs_tol=1e-6):
        id_image = key
    return id_image

  def __inspect_ann_video(self, ann_video_raw):
    # get id_video, ids_image, and num_images
    record = ann_video_raw[0]['task']['task_params']['record']
    id_video_real = record['attachment'].split('_')[-1][:-4].split('/')[0]
    id_image_to_timestamp_real = record['metadata']['additionalInfo']['framesTimestamp']
    num_images_real = len(ann_video_raw)
    ids_image_real = sorted(id_image_to_timestamp_real.keys(), key=int)
    assert ids_image_real[0] == '1' and ids_image_real[-1] == str(len(ids_image_real))

    errors = []
    anns_video_actor, anns_video_object = [], []
    for i, ann_image_raw in enumerate(ann_video_raw):
      # actor
      anns_image_actor_raw = ann_image_raw['task_result']['annotations'][0]['slotsChildren']
      anns_image_actor = [moma.Entity(ann_actor_raw, self.cn2en) for ann_actor_raw in anns_image_actor_raw]
      anns_video_actor += anns_image_actor

      # object
      anns_image_object_raw = ann_image_raw['task_result']['annotations'][1]['slotsChildren']
      anns_image_object = [moma.Entity(ann_object_raw, self.cn2en) for ann_object_raw in anns_image_object_raw]
      anns_video_object += anns_image_object

      # check id_video, ids_image, and num_images
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

    iids_video_actor = moma.sort(set([ann_video_actor.iid for ann_video_actor in anns_video_actor]))
    iids_video_object = moma.sort(set([ann_video_object.iid for ann_video_object in anns_video_object]))
    anns_instances_actor = [list(v) for _, v in itertools.groupby(anns_video_actor, lambda x: x.iid)]
    anns_instances_object = [list(v) for _, v in itertools.groupby(anns_video_object, lambda x: x.iid)]

    if not moma.is_consecutive(iids_video_actor):
      errors.append(f'[actor instance] iids not consecutive {iids_video_actor}')

    if not moma.is_consecutive(iids_video_object):
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

  def __inspect_ann_image(self, ann_image_raw):
    errors = []

    assert len(ann_image_raw['task_result']['annotations']) == 4

    """ actor & object """
    iids = []
    for i, type in enumerate(['actor', 'object']):
      assert self.cn2en[ann_image_raw['task_result']['annotations'][i]['label']] == type
      anns_entity_raw = ann_image_raw['task_result']['annotations'][i]['slotsChildren']
      anns_entity = [moma.Entity(ann_entity_raw, self.cn2en) for ann_entity_raw in anns_entity_raw]

      for ann_entity in anns_entity:
        # check type
        if ann_entity.type != type:
          errors.append(f'[{type}] wrong type {ann_entity.type}')

        # check cname
        taxonomy = self.taxonomy_actor if type == 'actor' else self.taxonomy_object
        if ann_entity.cname not in taxonomy:
          errors.append(f'[{type}] unseen cname {ann_entity.cname}')

        # check iid
        if not moma.is_entity(ann_entity.iid):
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
      assert self.cn2en[ann_image_raw['task_result']['annotations'][i+2]['label']] == type
      anns_description_raw = ann_image_raw['task_result']['annotations'][i+2]['slotsChildren']
      anns_description = [moma.Description(ann_description_raw, self.cn2en)
                          for ann_description_raw in anns_description_raw]

      for ann_description in anns_description:
        # check type
        if ann_description.type != type:
          errors.append(f'[{type}] wrong type {ann_description.type}')

        # check cname
        taxonomy = self.taxonomy_binary if type == 'binary description' else self.taxonomy_unary
        if ann_description.cname not in [x[0] for x in taxonomy]:
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
          if not moma.are_entities(iids_src+iids_trg):
            errors.append(f'[{type}] wrong iids_associated format {iids_src} -> {iids_trg}')
            continue

          if not set(iids_src+iids_trg).issubset(iids):
            errors.append(f'[{type}] unseen iids_associated {set(iids_src+iids_trg)} in {iids}')
            continue

          cnames_binary = [x[0] for x in self.taxonomy_binary]
          if ann_description.cname not in cnames_binary:  # unseen cname
            continue
          index = cnames_binary.index(ann_description.cname)
          type_src, type_trg = self.taxonomy_binary[index][1:]
          if (type_src == 'actor' and not moma.are_actors(iids_src)) or \
             (type_src == 'object' and not moma.are_objects(iids_src)) or \
             (type_src == 'actor/object' and not moma.are_entities(iids_src)) or \
             (type_trg == 'actor' and not moma.are_actors(iids_trg)) or \
             (type_trg == 'object' and not moma.are_objects(iids_trg)) or \
             (type_trg == 'actor/object' and not moma.are_entities(iids_trg)):
            errors.append(f'[{type}] wrong iids_associated {ann_description.iids_associated} '
                          f'for types {type_src} -> {type_trg}')

        elif type == 'unary description':
          iids_src = ann_description.iids_associated.split(',')
          if not moma.are_actors(iids_src):
            errors.append(f'[{type}] wrong iids_associated format {ann_description.iids_associated}')
            continue

          if not set(iids_src).issubset(iids):
            errors.append(f'[{type}] unseen iids_associated {set(iids_src)} in {iids}')

    return errors

  def inspect(self, verbose=True):
    for ann_video_raw in self.anns_video_raw:
      id_video = self.get_id_video(ann_video_raw)
      errors_video = self.__inspect_ann_video(ann_video_raw)
      if verbose and len(errors_video) > 0:
        msg = errors_video[0] if len(errors_video) == 1 else '; '.join(errors_video)
        print(f'Video {id_video}; {msg}')

      errors_image = []
      for ann_image_raw in ann_video_raw:
        id_image = self.get_id_image(ann_image_raw)
        errors_image += self.__inspect_ann_image(ann_image_raw)
        if verbose and len(errors_image) > 0:
          msg = errors_image[0] if len(errors_image) == 1 else '; '.join(errors_image)
          print(f'Video {id_video} Image {id_image}; {msg}')

      # error-free sub-activities
      if len(errors_video) == 0 and len(errors_image) == 0:
        self.iids_sact.append(id_video)

    print('\n ---------- REPORT (Phase 2) ----------')
    print(f'Number of sub-activity instances: {len(self.anns_video_raw)} -> {len(self.iids_sact)}')


def main():
  dir_moma = '/home/alan/ssd/moma'
  fname_ann_phase1 = 'video_anns_phase1_processed.json'
  fname_ann_phase2 = 'MOMA-videos-all.jsonl'

  ann_phase1 = AnnPhase1(dir_moma, fname_ann_phase1)
  ann_phase1.inspect(verbose=False)
  # distribution = ann_phase1.get_distribution()
  # pprint(distribution)

  ann_phase2 = AnnPhase2(dir_moma, fname_ann_phase2)
  ann_phase2.inspect(verbose=False)

  iids_sact_phase1 = set(itertools.chain(*ann_phase1.iid_act_to_iids_sact.values()))
  iids_sact_phase2 = set(ann_phase2.iids_sact)

  print(iids_sact_phase2-iids_sact_phase1)


if __name__ == '__main__':
  main()
