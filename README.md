# Learn and evaluate embeddings

0. First install requirements.txt

need to use python 2 environment for training embeddings (be more precise)

1. Collect data

You need to collect the data, and place it where the scripts expect it. This is described in the [here](https://github.com/clp-research/image_wac/tree/master/Data).

Note: we only need MSCOCO and SAIAPR

2. Train embeddings

(which environment?)
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

IN PYTHON2 environment:
go to ExtractFeats/sklearn-theano run python setup.py install
python 
-> extract features to create

  - saipr.npz # WORKS
  - mscoco.npz # DOESN'T WORK -> NOW WORKS with some modifications

...

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
- Train sit, ref, den embeddings
    - /home/users/lkopf/project/learn_eval_embeddings/026b_image_wac/Semantics/embeddings/train_embeddings_all_copy.py
- Train vis embeddings
    - /home/users/lkopf/project/learn_eval_embeddings/026b_image_wac/Semantics/embeddings/vis_av_2017_copy.py



3. Download embeddings
(- Sina embeddings)
- baroni web ebeddings
4. Download datasets for evaluation (vico)
- Datasets for evaluating embeddings (vico instructions)

- CIFAR-100 which is used for a zero-shot-like analysis
```
bash evaluation/vico/data/cifar100/download.sh
```
- **TODO**: remove 
-> saved here: /home/users/lkopf/project/learn_eval_embeddings/eval_embeds/data/datasets/cifar100/ 

- Data for Discriminative Attributes Task (SemEval 2018 Task 10) which is a word-only downstream task.
```
bash evaluation/vico/data/semeval_2018_10/download.sh
```
- **TODO**: remove 
-> saved here: /home/users/lkopf/project/learn_eval_embeddings/eval_embeds/data/datasets/semeval_2018_10
- VisualGenome (for unsupervised clustering analysis, TSNE plots)
    ```
    bash evaluation/vico/data/visualgenome/download.sh
    ```

5. convert txt embeddings to json and h5py for vico evaluation
    - **TODO**: change script in
    /home/users/lkopf/project/learn_eval_embeddings/vico_Laura/save_as_hdf5.py 

## Learn embeddings (will this ever work???)

* train_embeddings.py

 train different types of word embeddings from referring expressions

 running the script with the default settings will produce the following models:
 - w2v_sit_traindf_headonly_300dim.mod: situational embeddings (restricted to head nouns)
 - w2v_ref_traindf_300dim.mod: standard textual embeddings (predict left and right context)
 - w2v_den_traindf_window1_300dim.mod: denotational embeddings

* vis_av_2017.py

 computes visual embeddings for each word
 (averages over all the visual feature vectors of all positive instances of a word)
 outputs:
 - vis_av_refvocab_traindf.pklz

## Evaluate embeddings

### Zarrieß evaluation

in eval_embeds/ run
```
python evaluation/eval_embeddings.py 
```
If you want to save results as text file run:
```
python evaluation/eval_embeddings.py  | tee results/Zarriess_evaluation_results.txt
```

* eval_embeddings.py

 use this to run evaluation on some standard similarity benchmarks
 (correlations with human judgements, hypernymdirectionality)

* eval_embed_entail.py

 use this to evaluate embeddings on the approximate co-reference task

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
Plot tsne
```
python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_vis_pca_tsne
```

Add embedding name and according embedding dimension:
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

#### Zero-shot Analysis (delete)
preprocess embeddings: reduce dimensions to 100 dim.
```
python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_reduce_dim_embeds --embed_name='sit' --embed_dim=300
```

```
bash evaluation/vico/exp/cifar100/scripts/run.sh <gpu_id> <embed_name> <embed_dim> <num_held_out_classes>  <run_id>
```
--held_classes=20 --=0 --run=0

$ bash evaluation/vico/exp/cifar100/scripts/run.sh 0 20 0

<gpu_id> <run_id>

python -m evaluation.vico.exp.cifar100.run --exp exp_train --embed_name='sit' --embed_dim=300

#### Discriminative Attributes

extract words from embeddings:
```
python evaluation/vico/exp/semeval_2018_10/extract_embed_words.py 
```
run discriminative attributes
```
export CUDA_VISIBLE_DEVICES=0
```
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