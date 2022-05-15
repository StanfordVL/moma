#!/bin/bash

DIR_LOCAL="/media/ssd/moma"
DIR_REMOTE="/home/alanzluo/data/moma"
IP="34.69.154.32"

rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/anns/taxonomy/*.json" "$DIR_LOCAL/anns/taxonomy"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/anns/taxonomy/*.csv" "$DIR_LOCAL/anns/taxonomy"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/anns/split.json" "$DIR_LOCAL/anns"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/anns/split_fs.json" "$DIR_LOCAL/anns"

rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/videos/activity/" "$DIR_LOCAL/videos/activity"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/videos/activity_fr/" "$DIR_LOCAL/videos/activity_fr"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/videos/sub_activity/" "$DIR_LOCAL/videos/sub_activity"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/videos/sub_activity_fr/" "$DIR_LOCAL/videos/sub_activity_fr"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/videos/interaction/" "$DIR_LOCAL/videos/interaction"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/videos/interaction_frames/" "$DIR_LOCAL/videos/interaction_frames"
rsync -avznP --ignore-times alanzluo@"$IP:$DIR_REMOTE/videos/interaction_video/" "$DIR_LOCAL/videos/interaction_video"