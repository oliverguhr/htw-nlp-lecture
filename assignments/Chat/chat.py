import numpy as np
from termcolor import colored
from typing import List
import random
import pickle 
import os
import re, locale
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import tools
from tools import Pattern


tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
model = AutoModel.from_pretrained("bert-base-german-cased")

def encode(text):
    tokens = [tokenizer.encode(text, add_special_tokens=False)]

    input_ids = torch.tensor(tokens) 
    with torch.no_grad():
        all_hidden_states, _ = model(input_ids)
    
    # todo: implement a pooling strategy to generate a document vector
    # tip: take a look at slides from our last lesson
    document_vector = None #  <- you code :)

    assert(np.shape(document_vector) == (768,)) #  <- the output should have this shape
    return document_vector
    
def load_chat_texts():
    patterns = tools.load_file('dialog-ger.md')
    patterns.extend(tools.load_file('german-aixml.md'))
    patterns.extend(tools.load_file('german-aixml-2.md'))
    return patterns

def encode_chat_text_to_vectors(patterns):
    data = tools.load_if_exists("tmp.pickle")
    if data is not None:
        doc, doc_vecs, reponse_patterns = data
    else:
        doc = list()
        doc_vecs = list()
        reponse_patterns = list()        
        print("encoding sentences")
        for i,pattern in tqdm(enumerate(patterns), total=len(patterns)): 
            doc.extend(pattern.input)
            vectors = [encode(line) for line in pattern.input]                
            doc_vecs.extend(vectors)
            reponse_patterns.extend([pattern]*len(pattern.input))
        tools.save("tmp.pickle",[doc,doc_vecs,reponse_patterns])
    return doc, np.array(doc_vecs), reponse_patterns 


if __name__ == '__main__':
    
    # 1. load chat texts
    texts = load_chat_texts()
    # 2. convert texts into vectors
    doc, vectors, reponse_patterns  = encode_chat_text_to_vectors(texts)    

    topk = 5 # number of top scoring answers to print
    while True:
        query = input(colored('you: ', 'green'))
        query = query.strip().lower()
        query = re.sub(r'\W ', '', query) # remove non text chars
        query_vec = encode(query)
       
        # 3. compare user input to stored vectors unsing the dot product or cosine similarity    
        score = None # <- todo: write code to score the output here    
        topk_idx = None # <- todo: create a list with the [topk] document ids here 

        # 4. Output the answers with the highest score
        print('top %d texts similar to "%s"' % (topk, colored(query, 'green')))
        for idx in topk_idx:            
            matched_pattern = doc[idx]
            print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(matched_pattern, 'yellow')))       
                
        reponse_text = random.choice(reponse_patterns[topk_idx[0]].response)
        print(colored("robo: "+reponse_text+"\n","blue"))

        # 5. Create a chatbot startup :) 
