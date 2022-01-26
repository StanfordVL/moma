import ffmpeg
import json
import os
import shutil


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

  def select(self):
    fnames_raw = [ann['file_name'] for ann in self.anns]
    fnames_all_cn = os.listdir(os.path.join(self.dir_moma, 'videos/all'))
    fnames_all = [fname.split('_', 1)[1] for fname in fnames_all_cn]
    assert all([fname in fnames_all for fname in fnames_raw])
    fnames_raw_cn = [fnames_all_cn[fnames_all.index(fname)] for fname in fnames_raw]

    for fname_raw_cn, fname_raw in zip(fnames_raw_cn, fnames_raw):
      path_src = os.path.join(self.dir_moma, f'videos/all/{fname_raw_cn}')
      path_trg = os.path.join(self.dir_moma, f'videos/raw/{fname_raw}')
      shutil.copyfile(path_src, path_trg)

  @staticmethod
  def trim_video(file_src, file_trg, start, end):
    ffmpeg.input(file_src) \
          .video \
          .trim(start=start, end=end) \
          .setpts('PTS-STARTPTS') \
          .output(file_trg) \
          .run()

  @staticmethod
  def trim_image(file_src, file_trg, time):
    ffmpeg.input(file_src, ss=time) \
          .output(file_trg, vframes=1) \
          .run()

  def trim_act(self):
    for ann in self.anns:
      ann_act = ann['activity']
      file_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
      file_trg = os.path.join(self.dir_moma, 'videos/activity', f"{ann_act['key']}.mp4")
      start = ann_act['start_time']
      end = ann_act['end_time']
      self.trim_video(file_src, file_trg, start, end)

  def trim_sact(self):
    for ann in self.anns:
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        file_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
        file_trg = os.path.join(self.dir_moma, 'videos/sub_activity', f"{ann_sact['key']}.mp4")
        start = ann_sact['start_time']
        end = ann_sact['end_time']
        self.trim_video(file_src, file_trg, start, end)

  def trim_hoi(self):
    for ann in self.anns:
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        anns_hoi = ann_sact['higher_order_interactions']
        for ann_hoi in anns_hoi:
          file_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
          file_trg = os.path.join(self.dir_moma, 'videos/hoi_jpg', f"{ann_hoi['id']}.jpg")
          time = ann_hoi['time']
          self.trim_image(file_src, file_trg, time)
