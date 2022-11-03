import json
import os

path = os.path.join(os.getcwd(), 'data/embeddings/')

word_dim_dict = {'vis' : 1031, 'ref' : 300, 'den' : 300, 'sit' : 300, 'denref' : 600, 'baroni' : 400}

for name, dim in word_dim_dict.items():
    embed_path = os.path.join(path, f'{name}_{dim}dim_word_to_idx.json')
    
    print('Convert embeds:', embed_path)
    
    f = open(embed_path)
    data = json.load(f)
    word_list = list(data.keys())

    out_path = os.path.join(path, f'{name}_{dim}dim_words.json')
    print('Save embed_words:', out_path)
    
    with open(out_path, 'w') as outfile:
        json.dump(word_list, outfile)

print('Done!')
