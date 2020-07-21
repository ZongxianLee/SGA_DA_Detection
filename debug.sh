#!/bin/sh
CUDA_VISIBLE_DEVICES=6 python trainval_net_auto.py \
		   --bs 1 --cuda --net res101 \
 		   --dataset pascal_voc_0712 --dataset_t clipart \
	           --gc --lc --save_dir ./output --use_tfb

