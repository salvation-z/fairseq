# zh-en 

## prepare-data
position=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std
fairseq-preprocess --source-lang zh --target-lang en \
--trainpref $position/source/train_bpe30k \
--validpref $position/source/mt08_u8 \
--testpref $position/source/mt02_u8,$position/source/mt03_u8,$position/source/mt04_u8,$position/source/mt05_u8,$position/source/mt06_u8 \
--destdir $position/bin --workers 16

## prepare-data2
position=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std
fairseq-preprocess --source-lang zh --target-lang en \
--trainpref $position/source/train_bpe30k \
--validpref $position/source/mt02_u8 \
--testpref $position/source/mt08_u8,$position/source/mt03_u8,$position/source/mt04_u8,$position/source/mt05_u8,$position/source/mt06_u8 \
--destdir $position/bin2 --workers 16

## prepare-data3
position=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std
fairseq-preprocess --source-lang zh --target-lang en \
--trainpref $position/source/train_bpe30k \
--validpref $position/source/mt02_u8.30kbpe \
--testpref $position/source/mt08_u8.30kbpe,$position/source/mt03_u8.30kbpe,$position/source/mt04_u8.30kbpe,$position/source/mt05_u8.30kbpe,$position/source/mt06_u8.30kbpe \
--destdir $position/bin2 --workers 16

## train-baseline(example)
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    /home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/bin \
    --arch phrase_baseline \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir /home2/zhangzhuocheng/lab/translation/models/phrase/zh_en_baseline \
    --keep-last-epochs 5 \
    --log-format json

## train-baseline2(shaochenze++)

- vaswani_baseline: 
  - arch=transformer_wmt_en_de
  - model_dir=/home2/zhangzhuocheng/lab/translation/models/phrase/vaswani
  - data_dir=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/bin
- mine_baseline:
  - arch=phrase_baseline
  - model_dir=/home2/zhangzhuocheng/lab/translation/models/phrase/zh_en_baseline
  - data_dir=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/bin

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    $data_dir \
    --save-dir $model_dir \
    --arch $arch \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0007 \
    --min-lr 1e-09 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 2 \
    --no-progress-bar \
    --log-format json \
    --log-interval 10 \
    --save-interval-updates 1000 \
    --keep-interval-updates 5 \
    --keep-last-epochs 5 \
    --max-updates 15000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples

## Average

- baseline-vaswani: 
  - save_dir=/home2/zhangzhuocheng/lab/translation/models/phrase/vaswani
- baseline-mine: 
  - save_dir=/home2/zhangzhuocheng/lab/translation/models/phrase/zh_en_baseline

### 1. epoch
python average_checkpoints.py \
--inputs $save_dir/ \
--num-epoch-checkpoints 5 \
--output $save_dir/average_epoch.pt

### 2. update
python average_checkpoints.py \
--inputs $save_dir/ \
--num-update-checkpoints 5 \
--output $save_dir/average_update.pt


## Valid

- average_epoch:
  - model=average_epoch.pt
  - log=average_epoch.log.json
- average_update:
  - model=average_update.pt
  - log=average_update.log.json

- baseline-vaswani
  - model_dir=/home2/zhangzhuocheng/lab/translation/models/phrase/vaswani
  - data_dir=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/bin
- baseline-mine
  - model_dir=/home2/zhangzhuocheng/lab/translation/models/phrase/zh_en_baseline
  - data-dir=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/bin

### 1. generate
fairseq-generate $data_dir \
    --path $model_dir/$model \
    --beam 5 --remove-bpe \
    --results-path $model_dir/infer \
    --raw-output $model_dir/infer/raw.txt \
    | tee model_dir/$log

### 2. interactivate
- source:
  - source_dir=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/source
fairseq-interactive $data_dir \
    --path $model_dir/$model \
    --beam 5 --remove-bpe \
    --input $source_dir/mt02_u8.zh \
    --raw-output $model_dir/infer/out02.txt

## Score

- baseline-vaswani:
  - infer_dir=/home2/zhangzhuocheng/lab/translation/models/phrase/vaswani/infer
  - data_dir=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/source
- baseline-mine:
  - infer_dir=/home2/zhangzhuocheng/lab/translation/models/phrase/zh_en_baseline/infer
  - data_dir=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/source

perl scripts/multi-bleu.perl $data_dir/mt02_u8.en.low0 $data_dir/mt02_u8.en.low1 $data_dir/mt02_u8.en.low2 $data_dir/mt02_u8.en.low3 < $infer_dir/raw02.txt