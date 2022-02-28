import ffmpeg
import glob
import json
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
  def trim_image(path_src, path_trg, time):
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

  def trim_act(self, overwrite=False):
    paths_trg = []
    for ann in self.anns:
      ann_act = ann['activity']
      path_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
      path_trg = os.path.join(self.dir_moma, 'videos/activity', f"{ann_act['id']}.mp4")
      assert os.path.exists(path_src)
      if not os.path.exists(path_trg) or overwrite:
        start = ann_act['start_time']
        end = ann_act['end_time']
        self.trim_video(path_src, path_trg, start, end)
      paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/activity/*.mp4'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  def trim_sact(self, overwrite=False):
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
          self.trim_video(path_src, path_trg, start, end)
        paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/sub_activity/*.mp4'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  def trim_hoi(self, overwrite=False):
    paths_trg = []
    for ann in self.anns:
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        anns_hoi = ann_sact['higher_order_interactions']
        for ann_hoi in anns_hoi:
          path_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
          path_trg = os.path.join(self.dir_moma, 'videos/higher_order_interaction', f"{ann_hoi['id']}.jpg")
          assert os.path.exists(path_src)
          if not os.path.exists(path_trg) or overwrite:
            time = ann_hoi['time']
            self.trim_image(path_src, path_trg, time)
          paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/higher_order_interaction/*.jpg'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  def resize_act(self, overwrite=False):
    os.makedirs(os.path.join(self.dir_moma, 'videos/activity_sm'), exist_ok=True)

    paths_trg = []
    for ann in self.anns:
      ann_act = ann['activity']
      path_src = os.path.join(self.dir_moma, 'videos/activity', f"{ann_act['id']}.mp4")
      path_trg = os.path.join(self.dir_moma, 'videos/activity_sm', f"{ann_act['id']}.mp4")
      assert os.path.exists(path_src)
      if not os.path.exists(path_trg) or overwrite:
        self.resize_video(path_src, path_trg)
      paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/activity_sm/*.mp4'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)

  def resize_sact(self, overwrite=False):
    os.makedirs(os.path.join(self.dir_moma, 'videos/sub_activity_sm'), exist_ok=True)

    paths_trg = []
    for ann in self.anns:
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        path_src = os.path.join(self.dir_moma, 'videos/sub_activity', f"{ann_sact['id']}.mp4")
        path_trg = os.path.join(self.dir_moma, 'videos/sub_activity_sm', f"{ann_sact['id']}.mp4")
        assert os.path.exists(path_src)
        if not os.path.exists(path_trg) or overwrite:
          self.resize_video(path_src, path_trg)
        paths_trg.append(path_trg)

    paths_exist = glob.glob(os.path.join(self.dir_moma, 'videos/sub_activity_sm/*.mp4'))
    for path_exist in paths_exist:
      if path_exist not in paths_trg:
        os.remove(path_exist)
