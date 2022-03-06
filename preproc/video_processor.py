from collections import defaultdict
import ffmpeg
import glob
import json
import math
import os
import shutil
from torchvision import io


class VideoProcessor:
  """
   - all: unfiltered, untrimmed videos
   - raw: filtered, untrimmed videos
   - activity
   - sub-activity
   - higher-order interaction
  """
  def __init__(self, dir_moma):
    with open(os.path.join(dir_moma, 'anns/anns.json'), 'r') as f:
      anns = json.load(f)

    self.dir_moma = dir_moma
    self.anns = anns

  def select(self, overwrite=False):
    fnames_raw = [ann['file_name'] for ann in self.anns]
    fnames_all_cn = os.listdir(os.path.join(self.dir_moma, 'videos/all'))
    fnames_all = [fname.split('_', 1)[1] for fname in fnames_all_cn]
    assert all([fname in fnames_all for fname in fnames_raw])
    fnames_raw_cn = [fnames_all_cn[fnames_all.index(fname)] for fname in fnames_raw]

    paths_trg = []
    for fname_raw_cn, fname_raw in zip(fnames_raw_cn, fnames_raw):
      path_src = os.path.join(self.dir_moma, f'videos/all/{fname_raw_cn}')
      path_trg = os.path.join(self.dir_moma, f'videos/raw/{fname_raw}')
      assert os.path.exists(path_src)
      if not os.path.exists(path_trg) or overwrite:
        shutil.copyfile(path_src, path_trg)
      paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/raw/*.mp4'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  @staticmethod
  def trim_video(path_src, path_trg, start, end):
    ffmpeg.input(path_src) \
          .video \
          .trim(start=start, end=end) \
          .setpts('PTS-STARTPTS') \
          .output(path_trg) \
          .run()

  @staticmethod
  def sample_image(path_src, path_trg, time):
    ffmpeg.input(path_src, ss=time) \
          .output(path_trg, vframes=1) \
          .run()
    if not os.path.isfile(path_trg):  # strange bug: sometimes ffmpeg does not read the last frame!
      video = io.read_video(path_src, start_pts=time-1, pts_unit='sec')
      video = video[0][-1].permute(2, 0, 1)
      io.write_jpeg(video, path_trg, quality=50)

  @staticmethod
  def resize_video(path_src, path_trg, size=320):
    probe = ffmpeg.probe(path_src)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    if width < height:
      ffmpeg.input(path_src) \
        .video \
        .filter('scale', size, -2) \
        .output(path_trg) \
        .run()
    else:
      ffmpeg.input(path_src) \
        .video \
        .filter('scale', -2, size) \
        .output(path_trg) \
        .run()

  @staticmethod
  def trim_and_resize_video(path_src, path_trg, start, end, size=320):
    probe = ffmpeg.probe(path_src)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    if width < height:
      ffmpeg.input(path_src) \
        .video \
        .trim(start=start, end=end) \
        .setpts('PTS-STARTPTS') \
        .filter('scale', size, -2) \
        .output(path_trg) \
        .run()
    else:
      ffmpeg.input(path_src) \
        .video \
        .trim(start=start, end=end) \
        .setpts('PTS-STARTPTS') \
        .filter('scale', -2, size) \
        .output(path_trg) \
        .run()

  def trim_act(self, resize=False, overwrite=False):
    os.makedirs(os.path.join(self.dir_moma, 'videos/activity'), exist_ok=True)

    paths_trg = []
    for ann in self.anns:
      ann_act = ann['activity']
      path_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
      path_trg = os.path.join(self.dir_moma, 'videos/activity', f"{ann_act['id']}.mp4")
      assert os.path.exists(path_src)
      if not os.path.exists(path_trg) or overwrite:
        start = ann_act['start_time']
        end = ann_act['end_time']
        if resize:
          self.trim_video(path_src, path_trg, start, end)
        else:
          self.trim_and_resize_video(path_src, path_trg, start, end)
      paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/activity/*.mp4'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  def trim_sact(self, resize=False, overwrite=False):
    os.makedirs(os.path.join(self.dir_moma, 'videos/sub_activity'), exist_ok=True)

    paths_trg = []
    for ann in self.anns:
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        path_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
        path_trg = os.path.join(self.dir_moma, 'videos/sub_activity', f"{ann_sact['id']}.mp4")
        assert os.path.exists(path_src)
        if not os.path.exists(path_trg) or overwrite:
          start = ann_sact['start_time']
          end = ann_sact['end_time']
          if resize:
            self.trim_video(path_src, path_trg, start, end)
          else:
            self.trim_and_resize_video(path_src, path_trg, start, end)
        paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/sub_activity/*.mp4'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  def trim_hoi(self, duration=1, resize=False, overwrite=False):
    os.makedirs(os.path.join(self.dir_moma, 'videos/interaction_video'), exist_ok=True)

    paths_trg = []
    for i, ann in enumerate(self.anns):
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        anns_hoi = ann_sact['higher_order_interactions']
        anns_hoi = sorted(anns_hoi, key=lambda x: x['time'])
        for ann_hoi in anns_hoi:
          path_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
          path_trg = os.path.join(self.dir_moma, 'videos/interaction_video', f"{ann_hoi['id']}.mp4")
          assert os.path.exists(path_src)
          if not os.path.exists(path_trg) or overwrite:
            if ann_hoi['time']-duration/2 < 0:
              start = 0
              end = duration
            elif ann_hoi['time']+duration/2 > ann['duration']:
              end = ann['duration']
              start = end-duration
            else:
              start = ann_hoi['time']-duration/2
              end = start+duration
            assert math.isclose(end-start, duration, rel_tol=1e-4), \
                f"{ann_hoi['time']} -> [{start}, {end}) ({duration}s) from [0, {ann['duration']})"
            if resize:
              self.trim_video(path_src, path_trg, start, end)
            else:
              self.trim_and_resize_video(path_src, path_trg, start, end)
          paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/interaction_video/*.mp4'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  def sample_hoi(self, overwrite=False):
    os.makedirs(os.path.join(self.dir_moma, 'videos/interaction'), exist_ok=True)

    paths_trg = []
    for ann in self.anns:
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        anns_hoi = ann_sact['higher_order_interactions']
        for ann_hoi in anns_hoi:
          path_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
          path_trg = os.path.join(self.dir_moma, 'videos/interaction', f"{ann_hoi['id']}.jpg")
          assert os.path.exists(path_src)
          if not os.path.exists(path_trg) or overwrite:
            time = ann_hoi['time']
            self.sample_image(path_src, path_trg, time)
          paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/interaction/*.jpg'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  def sample_hoi_frames(self, num_frames=5, overwrite=False, split='test'):
    dir_out = os.path.join(self.dir_moma, 'videos/interaction_frames')
    os.makedirs(dir_out, exist_ok=True)
    assert num_frames%2 == 1  # odd number

    if os.path.exists(os.path.join(dir_out, 'timestamps.json')):
      with open(os.path.join(dir_out, 'timestamps.json'), 'r') as f:
        timestamps = json.load(f)
    else:
      timestamps = defaultdict(list)

    if split is not None:
      with open(os.path.join(self.dir_moma, 'anns/split.json'), 'r') as f:
        ids_act = json.load(f)[split]
      anns = [ann for ann in self.anns if ann['activity']['id'] in ids_act]
    else:
      anns = self.anns

    print(f'Number of activities: {len(anns)}')

    paths_trg = []
    for ann in anns:
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        anns_hoi = ann_sact['higher_order_interactions']
        anns_hoi = sorted(anns_hoi, key=lambda x: x['time'])
        for i, ann_hoi in enumerate(anns_hoi):
          path_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
          assert os.path.exists(path_src)

          now = ann_hoi['time']
          if i == 0:
            delta_right = (anns_hoi[i+1]['time']-now)/num_frames
            delta_left = delta_right
          elif i == len(anns_hoi)-1:
            delta_left = (now-anns_hoi[i-1]['time'])/num_frames
            delta_right = delta_left
          else:
            delta_left = (now-anns_hoi[i-1]['time'])/num_frames
            delta_right = (anns_hoi[i+1]['time']-now)/num_frames

          info = []
          for j in range(num_frames//2):
            # left
            id_hoi = f"{ann_hoi['id']}_l{j+1}"
            time = now-delta_left*(j+1)
            if time >= 0:
              info.append([id_hoi, time])
            # right
            id_hoi = f"{ann_hoi['id']}_r{j+1}"
            time = now+delta_right*(j+1)
            if time < ann['duration']:
              info.append([id_hoi, time])
          info = sorted(info, key=lambda x: x[1])

          timestamps[ann_hoi['id']] = info

          for id_hoi, time in info:
            path_trg = os.path.join(dir_out, f"{id_hoi}.jpg")
            if not os.path.exists(path_trg) or overwrite:
              self.sample_image(path_src, path_trg, time)
            paths_trg.append(path_trg)

    print('Done extracting frames')

    paths_exist = glob.glob(os.path.join(dir_out, '*.jpg'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

    with open(os.path.join(dir_out, 'timestamps.json'), 'w') as f:
      json.dump(timestamps, f, indent=2, sort_keys=True)
