#!/usr/bin/env sh
/home/yuchen/Library/caffe_train-master/build/tools/caffe train --solver=prototxt/pose_vgg_solver.prototxt --gpu=0 --weights=model/VGG_ILSVRC_19_layers.caffemodel  2>&1 | tee log/vggnet_$(date +%Y%m%d%H%M).txt
