#!/bin/bash

DIR_LOCAL="/home/alan/data/moma-lrg"
DIR_REMOTE="/home/alanzluo/data/moma"
IPS=("104.198.3.149" "34.132.24.89" "104.198.196.133" "34.134.180.176" "34.122.127.45")

for IP in "${IPS[@]}"
do
  rsync -avznP "$DIR_LOCAL/anns/taxonomy/*.json" alanzluo@"$IP":"$DIR_REMOTE/anns/taxonomy"
  rsync -avznP "$DIR_LOCAL/anns/taxonomy/*.csv" alanzluo@"$IP":"$DIR_REMOTE/anns/taxonomy"
  rsync -avznP "$DIR_LOCAL/anns/split.json" alanzluo@"$IP":"$DIR_REMOTE/anns"
  rsync -avznP "$DIR_LOCAL/anns/split_fs.json" alanzluo@"$IP":"$DIR_REMOTE/anns"

  rsync -avznP "$DIR_LOCAL/videos/activity/" alanzluo@"$IP":"$DIR_REMOTE/videos/activity"
  rsync -avznP "$DIR_LOCAL/videos/activity_fr/" alanzluo@"$IP":"$DIR_REMOTE/videos/activity_fr"
  rsync -avznP "$DIR_LOCAL/videos/sub_activity/" alanzluo@"$IP":"$DIR_REMOTE/videos/sub_activity"
  rsync -avznP "$DIR_LOCAL/videos/sub_activity_fr/" alanzluo@"$IP":"$DIR_REMOTE/videos/sub_activity_fr"
  rsync -avznP "$DIR_LOCAL/videos/interaction/" alanzluo@"$IP":"$DIR_REMOTE/videos/interaction"
  rsync -avznP "$DIR_LOCAL/videos/interaction_frames/" alanzluo@"$IP":"$DIR_REMOTE/videos/interaction_frames"
  rsync -avznP "$DIR_LOCAL/videos/interaction_video/" alanzluo@"$IP":"$DIR_REMOTE/videos/interaction_video"
done