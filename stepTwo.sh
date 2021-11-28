#!/bin/bash

python Test_main.py --maxdisp 192 \
					--model stackhourglass \
					--datapath "/mnt/Volume/SceneFlow" \
					--epochs 10 \
					--loadmodel "./trained/pretrained_model_KITTI2015.tar" \