set -eu

for name in CAMB CUUI AMU POST NTHU RAC UMC PKU SJTU UFC IPN IITB
do
    # echo ${name}
    CUDA_VISIBLE_DEVICES=0 python impara.py --src_file data/conll14/conll14_src.txt \
     --pred_file data/conll14/official_submissions/${name} \
     --restore_dir ${1}
done