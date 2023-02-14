#!/bin/sh

echo "Download Baroni embeddings..."

gdown https://drive.google.com/uc?id=1FCgl8GBgORa_UVOBbA10Ud5d9Un7_Ppm

echo "Extract embeddings..."

EMBED_PATH="./data/embeddings"
mkdir -p -- $EMBED_PATH

FILENAME="EN-wform.w.5.cbow.neg10.400.subsmpl.txt"
GZ_FILE="${PWD}/${FILENAME}.gz"
gunzip $GZ_FILE
EMBED="${PWD}/${FILENAME}"
NEW_EMBED="${EMBED_PATH}/baroni_400dim.txt"

mv $EMBED $NEW_EMBED

echo "Train situational, textual, and denotational embeddings..."
python ./training/train_sit-ref-den_embeddings.py

echo "Train visual and denref embeddings..."
python ./training/train_vis-denref_embeddings.py

echo "Preprocess embeddings..."
python ./training/preprocess_embeddings.py

echo "Done!"