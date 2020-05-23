import tempfile
from typing import Dict, Iterable, List, Tuple
from overrides import overrides

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

'''transformer stuff'''
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# from allennlp.modules.text_field_embedders import
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders import BertPooler




from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Auc
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate

# import the regularization
from allennlp.nn.regularizers import L2Regularizer, RegularizerApplicator

import pandas as pd
import os
import gc
from tqdm.auto import tqdm
import sys
sys.path.append("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/")
from src.preprocessing.text_preprocessing import preprocess_mimic

'''
We should throw out the X, where X is not good
'''

'''
Reimplementation of the AUC metric. However, we are simply not calling it correctly.
We need to actually do it in the forward pass
'''
import logging
logger = logging.getLogger(__name__)
logger.debug("hello")

@Model.register("MortalityClassifier")
class MortalityClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 regularizer_applicator: RegularizerApplicator = None
                 ):
        super().__init__(vocab, regularizer_applicator)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.auc = Auc()
        self.reg_app = regularizer_applicator

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor,
                metadata
                ) -> Dict[str, torch.Tensor]:

        # assert that metadata has the same length as the other ones. Then, they are parallel

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask) #horizontal; vertical (partial depth) might be good
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # reg_loss = self.get_regularization_penalty() # should not have to manually apply the regularization
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        self.accuracy(logits, label)
        preds = logits.argmax(-1)
        self.auc(preds, label)
        output = {'loss': loss, 'probs': probs, "metadata": metadata}
        print(f"we got some metadata{metadata}")
        return output

    '''this is called'''
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                "auc":self.auc.get_metric(reset)}

