# zh-en prep
position=/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std
fairseq-preprocess --source-lang zh --target-lang en --trainpref $position/source/train_bpe30k --validpref $position/source/mt08_u8 --testpref $position/source/mt02_u8,$position/source/mt03_u8,$position/source/mt04_u8,$position/source/mt05_u8,$position/source/mt06_u8 --destdir $position/bin --workers 10