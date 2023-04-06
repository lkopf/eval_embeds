# Learn and evaluate embeddings

 We examines various ways in which word embeddings from multi-modal contexts can be acquired. Adding different dimensions has the potential to exceed the limits on applying text data only. The embeddings we learn are based on the works of Zarrieß and Schlangen [2017](https://aclanthology.org/D17-1100) and are evaluated following the approach of Gupta et al. [2019](https://ieeexplore.ieee.org/document/9010420).

# Contents
- [Install Dependencies](#install-dependencies)
- [Collect Data](#collect-data)
- [Train Embeddings](#train-embeddings)
    - [Preprocess Data](#preprocess-data)
    - [Learn Embeddings](#learn-embeddings)
- [Evaluation](#evaluation)
    - [Similarity and Relatedness](#similarity-and-relatedness)
    - [Analogy Task](#analogy-task)
    - [Unsupervised Clustering Analysis](#unsupervised-clustering-analysis)
    - [Supervised Partitioning Analysis](#supervised-partitioning-analysis)
    - [Discriminative Attributes Task](#discriminative-attributes-task)
    - [Image Captioning](image-captioning)

# Install Dependencies

We provide two conda environment files for python 3 (`environment.yml`) and python 2 (`py2_environment.yml`). The files lists all dependencies which can easily be installed using the commands
```
conda env create -f environment.yml
```
and
```
conda env create -f py2_environment.yml
```

Once the installation is completed, launch the conda environment using
```
conda activate eval
```
or
```
conda activate py2
```

We will use the `eval` environment by default, unless indicated differently.

# Collect Data

Follow the instructions [here](https://github.com/lkopf/eval_embeds/tree/master/data) to collect the data and place them in the same given hierarchical structure in this folder

```
~/eval_embeds/data
```
Download Baroni embeddings
```
bash training/download_baroni.sh
```

# Train Embeddings

## Preprocess Data
For more detailed instructions see details on [preprocessing](https://github.com/lkopf/eval_embeds/tree/master/training/Preproc) and [extracting](https://github.com/lkopf/eval_embeds/tree/master/training/ExtractFeats).

Activate python 2 conda environment
```
conda activate py2
```
Install `sklearn-theano` following the instructions [here](http://sklearn-theano.github.io/install.html).

Compiling the referring expressions
```
python training/Preproc/preproc_refexps.py
```
Compiling the bounding boxes of the image regions of interest
```
python training/Preproc/preproc_region_defs.py
```
Extract heads from referring expressions, using POS-based patterns
```
python training/Preproc/refexp_heads.py
```
Extract features
```
python training/ExtractFeats/extract_feats.py
```

## Learn embeddings

Activate `eval` conda environment
```
conda activate eval
```
To learn all embeddings run
```
bash training/train_all_embeddings.sh
```

All embeddings can be found in:
```bash
~/eval_embeds/data/embeddings/
|-- baroni_400dim.txt # textual embeddings trained on large web corpus
|-- den_300dim.txt # denotational embeddings
|-- denref_600dim.txt # concatenated denotational and standard textual embeddings
|-- ref_300dim.txt # standard textual embeddings
|-- sit_300dim.txt # situational embeddings
|-- vis_1031dim.txt # visual embeddings
```

# Evaluation

Corresponding embedding names and dimensions. Useful for evaluating individual embeddings.

| embed_name | embed_dim |
|------------|-----------|
| 'baroni'   | 400       |
| 'den'      | 300       |
| 'denref'   | 600       |
| 'ref'      | 300       |
| 'sit'      | 300       |
| 'vis'      | 1031      |

## Similarity and Relatedness

Run evaluation on some standard similarity benchmarks (correlations with human judgements, hypernym directionality).
```
python evaluation/eval_embeddings.py 
```
If you want to save results as text file run:
```
python evaluation/eval_embeddings.py  | tee results/Zarriess_evaluation_results.txt
```

Notebook for qualitative similarity analysis can be found here:
```
./evaluation/qualitative_similarity.ipynb
```

## Analogy Task

Run analogy task
```
python evaluation/analogy_task.py
```
If you want to save results as text file run:
```
python evaluation/analogy_task.py | tee results/analogy_evaluation_results.txt
```

## Unsupervised Clustering Analysis

Download VisualGenome
```
bash evaluation/vico/data/visualgenome/download.sh
```

Run unsupervised clustering analysis
```
python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_unsupervised_clustering
```
If you want to save results as text file run:
```
python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_unsupervised_clustering  | tee results/unsupervised_clustering_evaluation_results.txt
```

Plot tsne for all embeddings:
```
bash evaluation/vico/exp/multi_sense_cooccur/exp_vis_pca_tsne_all_embeds.sh
```

Alternatively, plot tsne for single embedding, change embedding name and according embedding dimension:
```
python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_vis_pca_tsne --embed_name=<embed_name> --embed_dim=<embed_dim>
```

Results are saved in:
```
~/results/multi_sense_cooccur/{embed_name}
```

## Supervised Partitioning Analysis
Run supervised partitioning analysis
```
python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_supervised_partitioning
```
If you want to save results as text file run:
```
python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_supervised_partitioning  | tee results/supervised_partitioning_evaluation_results.txt
```

## Discriminative Attributes Task

Download SemEval 2018 Task 10
```
bash evaluation/vico/data/semeval_2018_10/download.sh
```
Extract words from embeddings
```
python evaluation/vico/exp/semeval_2018_10/extract_embed_words.py 
```
Run discriminative attributes for all embeddings.
```
export CUDA_VISIBLE_DEVICES=0
```
```
bash evaluation/vico/exp/semeval_2018_10/scripts/svm_all_embeds.sh
```
Alternatively, run single embedding:
```
bash evaluation/vico/exp/semeval_2018_10/scripts/svm_embeds.sh train_eval <embed_name> <embed_dim>
```
## Image Captioning

Create input files
```
python evaluation/Image_Captioning/create_input_files.py
```

Train with selected embeddings
```
python evaluation/Image_Captioning/train.py --pretrained_emb_name=<embed_name> --pretrained_emb_dim=<embed_dim>
```
Evaluate best trained model for all embeddings:
```
bash evaluation/Image_Captioning/eval_all_embeds.sh
```
Alternatively, run evaluation for single embedding:
```
python evaluation/Image_Captioning/eval.py --pretrained_emb_name=<embed_name>
```
Generate table of all results
```
python evaluation/Image_Captioning/all_scores_table.py
```
If you want to save results as text file run:
```
python evaluation/Image_Captioning/all_scores_table.py  | tee results/image_captioning_evaluation_results.txt
```
