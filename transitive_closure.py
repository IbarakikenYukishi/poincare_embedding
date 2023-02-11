import re
import pandas as pd
import os
from nltk.corpus import wordnet as wn
from tqdm import tqdm
try:
    wn.all_synsets
except LookupError as e:
    import nltk
    nltk.download('wordnet')

import nltk
nltk.download('omw-1.4')
# nltk.download('wordnet')


# make sure each edge is included only once
edge_set = set()
for synset in tqdm(wn.all_synsets(pos='n')):
    for hypernym in synset.closure(lambda s: s.hypernyms()):
        edge_set.add((synset.name(), hypernym.name()))
    for inst in synset.instance_hyponyms():
        for hypernym in inst.closure(lambda s: s.instance_hypernyms()):
            edge_set.add((inst.name(), hypernym.name()))
            for h in hypernym.closure(lambda s: s.hypernyms()):
                edge_set.add((inst.name(), h.name()))

nouns = pd.DataFrame(list(edge_set), columns=['id1', 'id2'])
nouns['weight'] = 1

# create datasets
os.makedirs("./dataset/wn_dataset/", exist_ok=True)
dataset_names = ["mammal", "solid", "tree", "worker", "adult", "leader", "instrument", "implement"]

for dataset_name in dataset_names:
    # Extract the set of nouns that have the word as a hypernym
    data_nodes = set(nouns[nouns.id2 == dataset_name + '.n.01'].id1.unique())
    data_nodes.add(dataset_name + '.n.01')
    # Select relations
    relations = nouns[nouns.id1.isin(data_nodes) & nouns.id2.isin(data_nodes)]
    relations.to_csv("./dataset/wn_dataset/" +
                     dataset_name + '_closure.csv', index=False)
