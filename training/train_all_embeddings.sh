#!/bin/sh

echo "Train situational, textual, and denotational embeddings..."
python ./training/train_sit-ref-den_embeddings.py

echo "Train visual and denref embeddings..."
python ./training/train_vis-denref_embeddings.py

echo "Preprocess embeddings..."
python ./training/preprocess_embeddings.py

echo "Done!"