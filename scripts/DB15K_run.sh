CUDA_VISIBLE_DEVICES=0 nohup python HiFi_main.py -dataset=DB15K \
  -batch_size=1024 \
  -margin=6 \
  -epoch=1000 \
  -dim=256 \
  -mu=0 \
  -save=./checkpoint/HiFi \
  -neg_num=128 \
  -learning_rate=1e-5 > result.txt &