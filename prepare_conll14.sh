#!/bin/bash
set -eu

wget https://www.comp.nus.edu.sg/~nlp/conll14st/official_submissions.tar.gz
tar -xvf official_submissions.tar.gz

wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar -xvf conll14st-test-data.tar.gz
cat conll14st-test-data/noalt/official-2014.combined.m2 | grep '^S' | cut -d ' ' -f 2- > conll14_src.txt
cp conll14_src.txt official_submissions/INPUT
cp conll14st-test-data/noalt/official-2014.combined.m2 ./

wget https://raw.githubusercontent.com/kanekomasahiro/gec_tutorial/main/src/convert_m2_to_parallel.py
python convert_m2_to_parallel.py conll14st-test-data/noalt/official-2014.0.m2 conll14_src.txt conll14_0_trg.txt
python convert_m2_to_parallel.py conll14st-test-data/noalt/official-2014.1.m2 conll14_src.txt conll14_1_trg.txt

rm official_submissions.tar.gz
rm conll14st-test-data.tar.gz
rm -r conll14st-test-data