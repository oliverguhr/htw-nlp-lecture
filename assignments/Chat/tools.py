import numpy as np
from termcolor import colored
from typing import List
import random
import pickle 
import os
import re, locale


class Pattern:    
    def __init__(self,name:str):
        self.input:List[str] = list()
        self.response:List[str] = list()
        self.name = name
        self.input_vectors: List[float] = None

def load_if_exists(filename:str):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None
def save(filename:str,data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_file(path):
    with open(path) as fp:
        patterns: List[Pattern] = list()
        pattern = None
        for line in fp:
            if line.startswith('##'):
                if pattern is not None: patterns.append(pattern)
                pattern = Pattern(line.replace('##','',1).strip())            
            elif line.startswith('*'):
                pattern.input.append(line.replace('*','',1).strip())
            elif line.startswith('  -'):
                if '*' not in line: # skips wildcard lines as we can't handel them yet
                    pattern.response.append(line.replace('  -','',1).strip())                    
        patterns.append(pattern)
        print(f'{len(patterns)} patterns loaded from file {path}')
        return patterns