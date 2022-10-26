OUTDIR=models/model
mkdir -p ${OUTDIR}
CUDA_VISIBLE_DEVICES=0 \
accelerate launch train_qe.py \
 --model_id bert-base-cased \
 --train_file data/train.tsv \
 --valid_file data/valid.tsv \
 --epochs 10 \
 --batch_size 32 \
 --outdir ${OUTDIR} 