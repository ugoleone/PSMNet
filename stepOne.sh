#!/bin/bash

python submission.py 	--maxdisp 192 \
			--model stackhourglass \
			--KITTI 2015 \
			--datapath "/mnt/Volume/KITTI/2015/testing/" \
			--loadmodel "./trained/pretrained_model_KITTI2015.tar" \
