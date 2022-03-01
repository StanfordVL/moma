#!/bin/bash

IPS=("104.198.3.149" "34.132.24.89" "104.198.196.133" "34.134.180.176" "34.122.127.45")

for IP in "${IPS[@]}"
do
  rsync -avzP /media/ssd/moma/anns/taxonomy/*.json alanzluo@"$IP":/home/alanzluo/data/moma/anns/taxonomy
  rsync -avzP /media/ssd/moma/anns/anns.json alanzluo@"$IP":/home/alanzluo/data/moma/anns
  rsync -avzP /media/ssd/moma/anns/anns_toy.json alanzluo@"$IP":/home/alanzluo/data/moma/anns
  rsync -avzP /media/ssd/moma/videos/activity/ alanzluo@"$IP":/home/alanzluo/data/moma/videos/activity
  rsync -avzP /media/ssd/moma/videos/activity_sm/ alanzluo@"$IP":/home/alanzluo/data/moma/videos/activity_sm
  rsync -avzP /media/ssd/moma/videos/sub_activity/ alanzluo@"$IP":/home/alanzluo/data/moma/videos/sub_activity
  rsync -avzP /media/ssd/moma/videos/sub_activity_sm/ alanzluo@"$IP":/home/alanzluo/data/moma/videos/sub_activity_sm
  rsync -avzP /media/ssd/moma/videos/higher_order_interaction/ alanzluo@"$IP":/home/alanzluo/data/moma/videos/higher_order_interaction
done