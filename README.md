# A BiLSTM + CRF based Chinese word segmenter

### Req

 PyTorch1.0, AllenNLP



### Run scripts:



export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=7

nohup python -u driver/TrainTest.py  --config_file config.ctb6.cfg > log 2>&1 &
tail -f log





### Performance

| CTB6 | P                 | R                 | F     |
| ---- | ----------------- | ----------------- | ----- |
| Dev  | 57828/59900=0.965 | 57828/59929=0.965 | 0.965 |
| Test | 78245/81273=0.963 | 78245/81579=0.959 | 0.961 |

### Speed

| CTB6 | Sentence num | Times  | Avg        |
| ---- | ------------ | ------ | ---------- |
| Dev  | 2078         | 10.89s | 190 sent/s |
| Test | 2795         | 14.16s | 199 sent/s |

