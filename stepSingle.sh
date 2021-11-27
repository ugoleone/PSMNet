#!/bin/bash

python Test_img.py 	--loadmodel "./trained/pretrained_model_KITTI2015.tar" \
			--leftimg "/mnt/Volume/SceneFlow/Driving/frames_cleanpass/15mm_focallength/scene_forwards/fast/left/0001.png" \
			--rightimg "/mnt/Volume/SceneFlow/Driving/frames_cleanpass/15mm_focallength/scene_forwards/fast/right/0001.png"
