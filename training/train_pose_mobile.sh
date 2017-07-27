#!/usr/bin/env sh
/home/yuchen/Library/modify/caffe_train-master/build/tools/caffe train --solver=prototxt/pose_mobile_solver.prototxt --gpu=0 --weights=model/mobilenet.caffemodel  2>&1 | tee log/MobileNet_$(date +%Y%m%d%H%M).txt
