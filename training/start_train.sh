#nohup sh train_pose_mobile.sh > MobileNet_modify_0.00004.log 2>&1 &

nohup sh train_pose_vgg.sh > Vggnet.log 2>&1 &

nohup sh train_pose_sque.sh > Sque.log 2>&1 &

#nohup sh train_pose_res.sh > Resnet.log 2>&1 &
