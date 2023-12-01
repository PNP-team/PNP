#!/bin/bash

# 定义函数，不断执行 git push 直到成功
function git_push {
    while true; do
        git push origin wenhao
        if [ $? -eq 0 ]; then
            echo "Push successful!"
            break
        else
            echo "Push failed. Retrying in 5 seconds..."
            sleep 5
        fi
    done
}

# 调用函数
git_push
