import glob
import json
import os
import pprint


def main():
  moma_dir = '/vision/u/zelunluo/moma'

  graph_anns_fname = 'graph_anns.json'
  video_anns_fname = 'video_anns_phase1_processed.json'
  trimmed_videos_dname = 'trimmed_videos_cn'
  videos_dname = 'videos_cn'

  with open(os.path.join(moma_dir, graph_anns_fname), 'r') as f:
    graph_anns = json.load(f)

  with open(os.path.join(moma_dir, video_anns_fname), 'r') as f:
    video_anns = json.load(f)

  trimmed_videos = [os.path.basename(x) for x in glob.glob(os.path.join(moma_dir, trimmed_videos_dname, '*.mp4'))]
  print(trimmed_videos[:10])
  key = list(video_anns.keys())[0]
  video_ann = video_anns[key]
  print(key)
  print(video_ann)
  print(video_ann.keys())
  pprint.pprint(video_ann['subactivity'])


if __name__ == '__main__':
  main()
