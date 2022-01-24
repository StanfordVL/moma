import ffmpeg
import json
import os
import shutil


class VideoProcessor:
  def __init__(self, dir_moma):
    with open(os.path.join(dir_moma, 'anns/anns.json'), 'r') as f:
      anns = json.load(f)

    self.dir_moma = dir_moma
    self.anns = anns

  def move(self):
    fnames_used = [ann['file_name'] for ann in self.anns]
    fnames_all_cn = os.listdir(os.path.join(self.dir_moma, 'videos/raw_all'))
    fnames_all = [fname.split('_', 1)[1] for fname in fnames_all_cn]
    assert all([fname in fnames_all for fname in fnames_used])
    fnames_used_cn = [fnames_all_cn[fnames_all.index(fname)] for fname in fnames_used]

    for fname, fname_cn in zip(fnames_used, fnames_used_cn):
      path_src = os.path.join(self.dir_moma, f'videos/raw_all/{fname_cn}')
      path_trg = os.path.join(self.dir_moma, f'videos/raw/{fname}')
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
      file_trg = os.path.join(self.dir_moma, 'videos/activities', f"{ann_act['key']}.mp4")
      start = ann_act['start_time']
      end = ann_act['end_time']
      self.trim_video(file_src, file_trg, start, end)

  def trim_sact(self):
    for ann in self.anns:
      anns_sact = ann['activity']['sub_activities']
      for ann_sact in anns_sact:
        file_src = os.path.join(self.dir_moma, 'videos/raw', ann['file_name'])
        file_trg = os.path.join(self.dir_moma, 'videos/sub_activities', f"{ann_sact['key']}.mp4")
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
          file_trg = os.path.join(self.dir_moma, 'videos/interactions', f"{ann_hoi['key']}.png")
          time = ann_hoi['time']
          self.trim_image(file_src, file_trg, time)
