#!/usr/bin/env sh
/home/yuchen/Library/modify/caffe_train-master/build/tools/caffe train --solver=prototxt/pose_sque_solver.prototxt --gpu=0 --weights=model/squeezenet_v1.1.caffemodel 2>&1 | tee log/sque_$(date +%Y%m%d%H%M).txt
