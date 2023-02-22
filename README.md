# Learn and evaluate embeddings

0. First install requirements.txt

1. Collect data

You need to collect the data, and place it where the scripts expect it. This is described in the [here](https://github.com/clp-research/image_wac/tree/master/Data).

Note: we only need MSCOCO and SAIAPR

2. Train embeddings

(which environment? py2???)
PREPROCESSING: https://github.com/clp-research/image_wac/tree/master/Preproc

- download image data for training Sina's embeddings:
https://github.com/clp-research/image_wac/tree/master/Data

- Compiling the referring expressions
python preproc_refexps.py # WORKS
-> creates one refdf (a Pandas DataFrame) for each of the refexp corpora, as well as json files for the splits.
  - PreProcOut/saiapr_berkeley_90-10_splits.json
  - PreProcOut/saiapr_refdf.pklz

- Compiling the bounding boxes of the image regions of interest
python preproc_region_defs.py
-> creates bbdfs, DataFrames containing bounding boxes, indexed by the same columns as used in the refdfs.
  - /Preproc/PreProcOut/mscoco_bbdf.pklz 
  - /Preproc/PreProcOut/saiapr_bbdf.pklz 

IN PYTHON2 environment: (py2 ???)
go to ExtractFeats/sklearn-theano run python setup.py install
python 
-> extract features to create

  - saipr.npz # WORKS
  - mscoco.npz # DOESN'T WORK -> NOW WORKS with some modifications


- preprocess data:
- for sit, ref, den
get refcoco_refdf.pklz etc as output (needed for refexp_heads.py)
- for vis:
get feature matrix saiapr.npz and mscoco.npz

- Generate inputs for sit embeddings
    - /home/users/lkopf/project/learn_eval_embeddings/026b_image_wac/Preproc/refexp_heads_copy.py
    - "this will extract heads from referring expressions, using POS-based patterns."
- Generate inputs for den & ref embeddings:
    - /home/users/lkopf/project/learn_eval_embeddings/026b_image_wac/Preproc/preproc_refexps_copy.py
- Generate inputs for vis embeddings
    - ???



3. Download datasets for evaluation (vico)
- Datasets for evaluating embeddings (vico instructions)

- CIFAR-100 which is used for a zero-shot-like analysis
```
bash evaluation/vico/data/cifar100/download.sh
```

- Data for Discriminative Attributes Task (SemEval 2018 Task 10) which is a word-only downstream task.
```
bash evaluation/vico/data/semeval_2018_10/download.sh
```

- VisualGenome (for unsupervised clustering analysis, TSNE plots)
    ```
    bash evaluation/vico/data/visualgenome/download.sh
    ```


## Learn embeddings (will this ever work???)

in 'eval' conda environment

run
```
bash training/train_all_embeddings.sh
```
in: ~/eval_embeds/training

Download Baroni embeddings.

- baroni_400dim.txt

All embeddings can be found in:
```
~/eval_embeds/data/embeddings/
```
- train_vis-denref_embeddings.py

    Train different types of word embeddings from referring expressions.

    Running the script with the default settings will produce the following models:
    - sit_300dim.txt: situational embeddings (restricted to head nouns)
    - ref_300dim.txt: standard textual embeddings (predict left and right context)
    - den_300dim.txt: denotational embeddings


- train_vis-denref_embeddings.py

    Computes visual embeddings for each word
    (averages over all the visual feature vectors of all positive instances of a word):
    - vis_1031dim.txt

    Concatenates denotational and referential embeddings:
    - denref_600dim.txt

- preprocess_embeddings.py

    Converts txt embeddings from .txt format to .json and .h5py formats for vico evaluation.

## Evaluate embeddings

### Zarrieß evaluation

with 'eval' conda env

- eval_embeddings.py

    Use this to run evaluation on some standard similarity benchmarks
    (correlations with human judgements, hypernym directionality).

in ~/eval_embeds/ run
```
python evaluation/eval_embeddings.py 
```
If you want to save results as text file run:
```
python evaluation/eval_embeddings.py  | tee results/Zarriess_evaluation_results.txt
```
#### START: NOT WORKING
Preprocess data for co-reference task:
```
python evaluation/make_pairdf.py
```
```
python evaluation/compile_featmats.py
```

- eval_embed_entail.py

    Use this to evaluate embeddings on the approximate co-reference task.

```
python evaluation/eval_embed_entail.py
```
#### END: NOT WORKING

### Vico evaluation

#### Unsupervised Clustering Analysis
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
python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_vis_pca_tsne --embed_name='sit' --embed_dim=300
```

Corresponding embedding names and dimensions.

| embed_name | embed_dim |
|------------|-----------|
| 'vis'      | 1031      |
| 'ref'      | 300       |
| 'den'      | 300       |
| 'sit'      | 300       |
| 'denref'   | 600       |
| 'baroni'   | 400       |

Results are saved in:

~/results/multi_sense_cooccur/{embed_name}

#### Discriminative Attributes

extract words from embeddings:
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
bash evaluation/vico/exp/semeval_2018_10/scripts/svm_embeds.sh train_eval sit 300
```
#### Image Captioning

adapt create_input_files.py to include paths to COCO data:
* karpathy_json_path
* image_folder
in train.py change:
* pretrained_emb_name
+ pretrained_emb_dim
also in eval.py adapt folders:
* pretrained_embeddings_name -> TODO: can I make this adaptable outside script?

1) Create input files
```
python evaluation/Image_Captioning/create_input_files.py
```

2) Train with selected embeddings
```
python evaluation/Image_Captioning/train.py 
```
(takes forever...)

3) Eval best trained model
```
python evaluation/Image_Captioning/eval.py 
```

* TODO: cite sources for integration of metrics:
- https://github.com/ruizhao1997/a-PyTorch-Tutorial-to-Image-Captioning
- https://github.com/salaniz/pycocoevalcap

4) Generate table of all results
```
python results/Image-Captioning/all_scores_table.py
```
If you want to save results as text file run:
```
python results/Image-Captioning/all_scores_table.py  | tee results/image_captioning_evaluation_results.txt
```
### Data structure

```
eval_embeds/
|-- data
    |-- embeddings
    |.. datasets
|-- train ???
    |-- ???
|-- evaluation
    |-- Image-Captioning
    |-- vico
|-- results
```