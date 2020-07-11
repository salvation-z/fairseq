# zh-en 

## prepare-data
position=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std
fairseq-preprocess --source-lang zh --target-lang en --trainpref $position/source/train_bpe30k --validpref $position/source/mt08_u8 --testpref $position/source/mt02_u8,$position/source/mt03_u8,$position/source/mt04_u8,$position/source/mt05_u8,$position/source/mt06_u8 --destdir $position/bin --workers 10

## train
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
    --keep-last-epochs 5