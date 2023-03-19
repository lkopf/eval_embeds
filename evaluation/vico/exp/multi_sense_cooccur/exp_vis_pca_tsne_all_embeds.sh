#!/bin/bash

declare -A all_embeds=( ["vis"]=1031 ["ref"]=300 ["den"]=300 ["sit"]=300 ["denref"]=600 ["baroni"]=400 )

for key in "${!all_embeds[@]}"
do
    python -m evaluation.vico.exp.multi_sense_cooccur.run --exp exp_vis_pca_tsne --embed_name=$key --embed_dim=${all_embeds[$key]};
done

echo "Done!"