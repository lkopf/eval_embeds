#!/bin/bash

declare -A all_embeds=( ["vis"]=1031 ["ref"]=300 ["den"]=300 ["sit"]=300 ["denref"]=600 ["baroni"]=400 )

for key in "${!all_embeds[@]}"
do
    bash evaluation/vico/exp/semeval_2018_10/scripts/svm_embeds.sh train_eval $key ${all_embeds[$key]}
done

echo "Done!"

