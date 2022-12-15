# IMPARA

This is an UNOFFICIAL implementation of IMPARA, one of the reference-less metric for Grammatical Error Correction, proposed in the following paper:

```
@inproceedings{maeda-etal-2022-impara,
    title = "{IMPARA}: Impact-Based Metric for {GEC} Using Parallel Data",
    author = "Maeda, Koki  and
      Kaneko, Masahiro  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.316",
    pages = "3578--3588",
}
```

# Trained Quallity Estimation model

I have uploaded the trained Quality Estimation model for IMPARA to Huggingface Hub.  
The id of the model card is `gotutiyan/IMPARA-QE`. You can use it by `BertForSequenceClassification.from_pretrained('gotutiyan/IMPARA-QE')`.

`gotutiyan/IMPARA-QE` achieves 95.93 for Peason's correlation and 93.01 for Spearman's (with 'bert-base-cased' for SE model). For more information, please see [here](https://github.com/gotutiyan/IMPARA#correlation-with-human-evaluation).  
Note that this results does not fully achieve the results of the paper.

# Usage

### CLI

If you don't specify `--restore_dir`, `gotutiyan/IMPARA-QE` will be used for the QE model.

```sh
python impara.py \
 --src <source_file> \
 --pred <prediction_file>

# If you use your custom QE model
# python impara.py \
#  --src <source_file> \
#  --pred <prediction_file> \
#  --restore_dir <directory of your custom QE model>
```

### API
```python
from transformers import AutoTokenizer, BertForSequenceClassification
from modeling import IMPARA, SimilarityEstimator
QE_model_id = 'gotutiyan/IMPARA-QE'
se_model = SimilarityEstimator('bert-base-cased')
qe_model = BertForSequenceClassification.from_pretrained(QE_model_id)
impara = IMPARA(se_model, qe_model, threshold=0.9)
tokenizer = AutoTokenizer.from_pretrained(QE_model_id)
s_encode = tokenizer('This is sample sentence.', return_tensors='pt')
p_encode = tokenizer('This is a sample sentence.', return_tensors='pt')
scores = impara(
    src_input_ids=s_encode['input_ids'],
    src_attention_mask=s_encode['attention_mask'],
    pred_input_ids=p_encode['input_ids'],
    pred_attention_mask=p_encode['attention_mask'],
)
print(scores) # tensor([0.9617], grad_fn=<IndexPutBackward0>)
```

# Experiments Procedure

Confirmed that it works on python 3.8.10.

### 1. Install

Maybe `requirements.txt` contains unrelated modules but necessary modules are included.

```sh
pip install -r requirements.txt
```

### 2. Prepare data
```sh
mkdir data
cd data
wget https://www.comp.nus.edu.sg/~nlp/conll13st/release2.3.1.tar.gz
tar -xf release2.3.1.tar.gz 
git clone https://github.com/Jason3900/M2Convertor.git
python M2Convertor/conv_m2.py \
 -f release2.3.1/original/data/official-preprocessed.m2 \
 -p release2.3.1/original/data/conll13
```
We will use `release2.3.1/original/data/conll13.src` and `release2.3.1/original/data/conll13.trg`.

### 3. Create supervision data for quality estimation model
Create supervison data from CoNLL-13 parallel data.

The script tries to create 30 samples for each parallel data, so we temporarily obtain about 40000 supervision instances (about 1380 sentences x 30 samples). Then, the instances are shuffled randomly and used only 4096 instances from the front.

The paper said the data is divided 8:1:1 for train, valid and test set, so I divided 3276:410:410.

```sh
pwd # IMPARA/data
python ../create_data_for_qe.py \
 --src release2.3.1/original/data/conll13.src \
 --trg release2.3.1/original/data/conll13.trg \
 > all.tsv

cat all.tsv | awk 'NR==1,NR==410 {print}' > test.tsv
cat all.tsv | awk 'NR==411,NR==820 {print}' > valid.tsv
cat all.tsv | awk 'NR>=821 {print}' > train.tsv
```

### 4. Train the quality estimation model
Please rewrite OUTDIR variable to be appropriate.
Since the training data is very small, the training will be finished around 10 minutes on a A100.

This script also works on multiple GPUs with Accelerate module of Huggingface but a single GPU is enough to train. Here is a setting of Aceelerate to train on a single GPU:

```sh
accelerate config
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
# Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 0
# Do you want to run your training on CPU only (even if a GPU is available)? [yes/NO]:NO
# Do you want to use DeepSpeed? [yes/NO]: NO
# What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
# Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: NO
```
Then, 
```sh
pwd # IMPARA/
bash train_qe.sh

# OUTDIR=models/model
# mkdir -p ${OUTDIR}
# CUDA_VISIBLE_DEVICES=0 \
# accelerate launch train_qe.py \
#  --train_file data/train.tsv \
#  --valid_file data/valid.tsv \
#  --epochs 10 \
#  --batch_size 32 \
#  --outdir ${OUTDIR} 
```

The results will be saved as the following format.
```
models/model
├── best
│   ├── config.json
│   ├── impara_config.json
│   ├── pytorch_model.bin
├── last
│   ├── config.json
│   ├── impara_config.json
│   └── pytorch_model.bin
└── log.json
```

### 5. Evaluate
The way is the same as `Usage` section mentioned above.

Here is an example if your trained model is saved in `models/model/best`.
```sh
python impara.py \
 --src <source_file> \
 --pred <prediction_file> \
 --restore_dir models/model/best
```

# Correlation with Human Evaluation

Here is an example to compute correlation with [Grundkiewicz +15](https://aclanthology.org/D15-1052/)'s Expected Wins score.

```sh
mkdir data/conll14
cd data/conll14
bash ../../prepare_conll14.sh
cd ../../
bash conll14_score.sh path/to/model > result.txt
python correlation.py --human Grundkiewicz15_EW.txt --system result.txt
```

The input of the `correleation.py` is 12 lines consisting of `CAMB CUUI AMU POST NTHU RAC UMC PKU SJTU UFC IPN IITB` scores.

The trained QE model of `gotutiyan/IMPARA-QE` achieves 95.93 for Peason's correlation and 93.01 for Spearman's.