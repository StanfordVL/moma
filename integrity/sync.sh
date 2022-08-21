#!/bin/bash

DIR_LOCAL="/home/alan/data/moma-lrg"
DIR_REMOTE="/home/alanzluo/data/moma"
IP="34.138.83.196"

rsync -avznP alanzluo@"$IP:$DIR_REMOTE/anns/taxonomy/*.json" "$DIR_LOCAL/anns/taxonomy"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/anns/taxonomy/*.csv" "$DIR_LOCAL/anns/taxonomy"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/anns/split.json" "$DIR_LOCAL/anns"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/anns/split_fs.json" "$DIR_LOCAL/anns"

rsync -avznP alanzluo@"$IP:$DIR_REMOTE/videos/activity/" "$DIR_LOCAL/videos/activity"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/videos/activity_fr/" "$DIR_LOCAL/videos/activity_fr"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/videos/sub_activity/" "$DIR_LOCAL/videos/sub_activity"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/videos/sub_activity_fr/" "$DIR_LOCAL/videos/sub_activity_fr"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/videos/interaction/" "$DIR_LOCAL/videos/interaction"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/videos/interaction_frames/" "$DIR_LOCAL/videos/interaction_frames"
rsync -avznP alanzluo@"$IP:$DIR_REMOTE/videos/interaction_video/" "$DIR_LOCAL/videos/interaction_video"