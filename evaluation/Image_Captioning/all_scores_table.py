import json
import pandas as pd

embed_list = ['vis', 'ref', 'den', 'sit', 'denref', 'baroni']

results = []

for embed in embed_list:
    file = open(f'./results/Image_Captioning/score_dict_{embed}.json', 'r')
    embed_scores = json.load(file)
    select_scores = embed_scores.copy()
    [select_scores.pop(key) for key in ['Bleu_2', 'Bleu_3']]
    results.append([embed]+["%.3f"%c for c in select_scores.values()])

df = pd.DataFrame(results,columns=['Model', 'Bleu_1', 'Bleu_4', 'CIDEr', 'SPICE'])
print(df.to_latex(index=False))
