CUDA_VISIBLE_DEVICES=2,3 python PCB.py -d market -a resnet50 -b 64 -j 4 --log logs/market-1501/PCB_20epoch/  --feature 256 --height 384 --width 128 --epochs 20 --step-size 20 --data-dir ~/datasets/Market-1501/


CUDA_VISIBLE_DEVICES=2,3 python RPP.py -d market -a resnet50_rpp -b 64 -j 4 --log logs/market-1501/RPP/  --feature 256 --height 384 --width 128 --epochs 50 --step_size 20 --data-dir ~/datasets/Market-1501/ --resume logs/market-1501/PCB_20epoch/checkpoint.pth.tar
