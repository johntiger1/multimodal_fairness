import tempfile
from typing import Dict, Iterable, List, Tuple

import torch

import allennlp
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate

import pandas as pd
import os
class MortalityReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 listfile: str = None,
                 notes_dir: str = None,
                 ):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.listfile = listfile
        self.notes_dir = notes_dir

    def _read(self, file_path: str) -> Iterable[Instance]:
        '''Expect: one instance per line'''
        with open(self.listfile, "r") as file:
            for line in file:
                info_filename, label = line.split(",")
                info = info_filename.split("_")
                patient_id = info[0]
                eps = ("".join([c for c in info[1] if c.isdigit()]))
                notes = pd.read_pickle(os.path.join(self.notes_dir, patient_id))

                text = notes[notes["EPISODES"] == eps]["text"]
                tokens = self.tokenizer.tokenize(text)

                text_field = TextField(tokens, self.token_indexers)
                label_field = LabelField(label)
                fields = {'text': text_field, 'label': label_field}
                yield Instance(fields)


        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                tokens = self.tokenizer.tokenize(text)
                if self.max_tokens:
                    tokens = tokens[:self.max_tokens]
                text_field = TextField(tokens, self.token_indexers)
                label_field = LabelField(sentiment)
                fields = {'text': text_field, 'label': label_field}
                yield Instance(fields)