#!/bin/bash

python Test_Kitti.py --maxdisp 192 \
					--model stackhourglass \
                    --datatype 2015 \
					--datapath "/mnt/Volume/KITTI/2015/testing/" \
					--epochs 300 \
					--loadmodel "./trained/pretrained_model_KITTI2015.tar" \
                    --savemodel "." \