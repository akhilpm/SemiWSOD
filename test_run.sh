#!/bin/bash

for i in {1..20}
do
	echo "==============" "EPOCH" $i "=============="
	python run.py test --net vgg16 --dataset voc_2007_test --session 1 --epoch $i --cuda -ap color_mode=RGB image_range=1 mean="[0.485, 0.456, 0.406]" std="[0.229, 0.224, 0.225]"
done


# ./test_run.sh | tee output.txt
