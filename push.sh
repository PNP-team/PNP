#!/bin/bash

while true
do
    git push -u origin main
    sleep 60  # 每隔60秒执行一次，你可以根据需要调整间隔时间
done
