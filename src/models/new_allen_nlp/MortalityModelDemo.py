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
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Auc
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate

import pandas as pd
import os
import gc
'''
We should throw out the X, where X is not good
'''

'''
Reimplementation of the AUC metric. However, we are simply not calling it correctly.
We need to actually do it in the forward pass
'''
# class Auc(Metric):
#     """
#     The AUC Metric measures the area under the receiver-operating characteristic
#     (ROC) curve for binary classification problems.
#     """
#
#     def __init__(self, positive_label=1):
#         super().__init__()
#         self._positive_label = positive_label
#         self._all_predictions = torch.FloatTensor()
#         self._all_gold_labels = torch.LongTensor()
#
#     def __call__(
#         self,
#         predictions: torch.Tensor,
#         gold_labels: torch.Tensor,
#         mask: Optional[torch.BoolTensor] = None,
#     ):
#         """
#         # Parameters
#         predictions : `torch.Tensor`, required.
#             A one-dimensional tensor of prediction scores of shape (batch_size).
#         gold_labels : `torch.Tensor`, required.
#             A one-dimensional label tensor of shape (batch_size), with {1, 0}
#             entries for positive and negative class. If it's not binary,
#             `positive_label` should be passed in the initialization.
#         mask : `torch.BoolTensor`, optional (default = None).
#             A one-dimensional label tensor of shape (batch_size).
#         """
#
#         predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
#
#         # Sanity checks.
#         if gold_labels.dim() != 1:
#             raise ConfigurationError(
#                 "gold_labels must be one-dimensional, "
#                 "but found tensor of shape: {}".format(gold_labels.size())
#             )
#         if predictions.dim() != 1:
#             raise ConfigurationError(
#                 "predictions must be one-dimensional, "
#                 "but found tensor of shape: {}".format(predictions.size())
#             )
#
#         unique_gold_labels = torch.unique(gold_labels)
#         if unique_gold_labels.numel() > 2:
#             raise ConfigurationError(
#                 "AUC can be used for binary tasks only. gold_labels has {} unique labels, "
#                 "expected at maximum 2.".format(unique_gold_labels.numel())
#             )
#
#         gold_labels_is_binary = set(unique_gold_labels.tolist()) <= {0, 1}
#         if not gold_labels_is_binary and self._positive_label not in unique_gold_labels:
#             raise ConfigurationError(
#                 "gold_labels should be binary with 0 and 1 or initialized positive_label "
#                 "{} should be present in gold_labels".format(self._positive_label)
#             )
#
#         if mask is None:
#             batch_size = gold_labels.shape[0]
#             mask = torch.ones(batch_size, device=gold_labels.device).bool()
#
#         self._all_predictions = self._all_predictions.to(predictions.device)
#         self._all_gold_labels = self._all_gold_labels.to(gold_labels.device)
#
#         self._all_predictions = torch.cat(
#             [self._all_predictions, torch.masked_select(predictions, mask).float()], dim=0
#         )
#         self._all_gold_labels = torch.cat(
#             [self._all_gold_labels, torch.masked_select(gold_labels, mask).long()], dim=0
#         )
#
#     def get_metric(self, reset: bool = False):
#         if self._all_gold_labels.shape[0] == 0:
#             return 0.5
#         false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
#             self._all_gold_labels.cpu().numpy(),
#             self._all_predictions.cpu().numpy(),
#             pos_label=self._positive_label,
#         )
#         auc = metrics.auc(false_positive_rates, true_positive_rates)
#         if reset:
#             self.reset()
#         return auc
#
#     @overrides
#     def reset(self):
#         self._all_predictions = torch.FloatTensor()
#         self._all_gold_labels = torch.LongTensor()

class MortalityReader(DatasetReader):
    def __init__(self,
                 lazy: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = 512,
                 listfile: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
                 notes_dir: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes",
                 ):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.listfile = listfile
        self.notes_dir = notes_dir

    def get_stats(self, file_path: str):
        '''

        '''
        # get stats on the dataset listed at _path_
        from collections import defaultdict
        self.stats = defaultdict(int)

        with open(file_path, "r") as file:
            file.readline() # could also pandas readcsv and ignore first line
            for line in file:
                info_filename, label = line.split(",")
                self.stats[int(label)] +=1
        return self.stats

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        '''Expect: one instance per line'''
        with open(file_path, "r") as file:
            file.readline() # could also pandas readcsv and ignore first line
            for line in file:
                info_filename, label = line.split(",")
                info = info_filename.split("_")
                patient_id = info[0]
                eps = int("".join([c for c in info[1] if c.isdigit()]))
                notes = pd.read_pickle(os.path.join(self.notes_dir, patient_id, "notes.pkl"))

                if len(notes[notes["EPISODES"] == eps]) > 0:
                    text = notes[notes["EPISODES"] == eps]["TEXT"].iloc[0]
                    # join the texts together, or simply use the first one
                    tokens = self.tokenizer.tokenize(text)

                    text_field = TextField(tokens, self.token_indexers)
                    label_field = LabelField(label)
                    fields = {'text': text_field, 'label': label_field}
                    yield Instance(fields)



class MortalityClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.auc = Auc()

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        self.accuracy(logits, label)
        preds = logits.argmax(-1)
        self.auc(preds, label)
        output = {'loss': loss, 'probs': probs}
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                "auc":self.auc.get_metric(reset)}
#
#
def build_dataset_reader() -> DatasetReader:
    return MortalityReader(lazy=True)
#

# "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv"
'''
The issue is that this may for somereason read everything into memory
'''
def read_data(
    reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv")
    validation_data = reader.read("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv")
    return training_data, validation_data


# def read_data(
#     reader: DatasetReader
# ) -> Tuple[Iterable[Instance], Iterable[Instance]]:
#     print("Reading data")
#     training_data = reader.read("quick_start/data/movie_review/train.tsv")
#     validation_data = reader.read("quick_start/data/movie_review/dev.tsv")
#     return training_data, validation_data

#
def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)
#

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    # turn the tokens into 300 dim embedding. Then, turn the embeddings into encodings
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=300, num_embeddings=vocab_size)})
    encoder = CnnEncoder(embedding_dim=300, ngram_filter_sizes = (2,3,4,5),
                         num_filters=5) # num_filters is a tad bit dangerous: the reason is that we have this many filters for EACH ngram f

    # the output dim is just the num filters *len(ngram_filter_sizes)
    return MortalityClassifier(vocab, embedder, encoder)
#
def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset, batch_size=64
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader


def build_data_loaders_from_reader(dataset_reader, vocab, batch_size=64):
    train_data, dev_data = read_data(dataset_reader)

    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader

#
def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
        cuda_device=0
    )
    return trainer

'''
we pass in the vocab to ensure that we are speaking the same language!
'''
def run_training_loop(model, dataset_reader, vocab, use_gpu=False, batch_size =32):
    # move the model over, if necessary, and possible
    gpu_device = torch.device("cuda:0" if use_gpu  else "cpu")
    model = model.to(gpu_device)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    gc.collect()

    train_loader, dev_loader = build_data_loaders_from_reader(dataset_reader,vocab, batch_size=batch_size)
    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this course, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
        trainer.train()
    del train_loader
    del dev_loader
    gc.collect()
    return model, dataset_reader

def main():

    args = lambda x: None
    args.batch_size = 64
    import time

    start_time = time.time()
    # mr = MortalityReader()
    # instances = mr.read("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv")
    # for inst in instances[:10]:
    #     print(inst)
    print("we are running with the following info")
    print("Torch version {} Cuda version {} cuda available? {}".format(torch.__version__, torch.version.cuda,
                                                                       torch.cuda.is_available()))
    # We've copied the training loop from an earlier example, with updated model
    # code, above in the Setup section. We run the training loop to get a trained
    # model.

    dataset_reader = build_dataset_reader()

    dataset_reader.get_stats("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv")
    for key in sorted(dataset_reader.stats.keys()):
        print("{} {}".format(key,dataset_reader.stats[key]))
    dataset_reader.get_stats("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv")

    for key in sorted(dataset_reader.stats.keys()):
        print("{} {}".format(key,dataset_reader.stats[key]))
    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)
    vocab = build_vocab(train_data + dev_data)
    del train_data
    del dev_data
    model = build_model(vocab)

    model, dataset_reader = run_training_loop(model,dataset_reader, vocab, use_gpu=True, batch_size=args.batch_size)

    # Now we can evaluate the model on a new dataset.
    test_data = dataset_reader.read(
        '/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv')
    test_data.index_with(model.vocab)
    data_loader = DataLoader(test_data, batch_size=args.batch_size)

    # results = evaluate(model, data_loader, -1, None)
    # print(results)

    # will cause an exception due to outdated cuda driver? Not anymore!
    results = evaluate(model, data_loader, 0, None)

    print("we succ fulfilled it")
    with open("nice_srun_time.txt", "w") as file:
        file.write("it is done\n{}\nTook {}".format(results, time.time() - start_time))





if __name__ == __name__:
    main()