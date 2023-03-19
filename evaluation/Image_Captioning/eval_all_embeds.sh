#!/bin/bash

# all_embeds=("vis" "ref" "den" "sit" "denref" "baroni")
all_embeds=("ref" "den" "denref" "baroni")

for embed in "${all_embeds[@]}"; do
    python evaluation/Image_Captioning/eval.py --pretrained_emb_name=$embed
done

echo "Done!"