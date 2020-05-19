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
from allennlp.modules.seq2vec_encoders import BertPooler

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
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("hello")

class MortalityReader(DatasetReader):
    def __init__(self,
                 lazy: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = 512,
                 listfile: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
                 notes_dir: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes",
                 skip_patients_file: str ="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/null_patients.txt"
                 ):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.listfile = listfile
        self.notes_dir = notes_dir

        self.null_patients = []
        with open(skip_patients_file, "r") as file:
            for line in file:
                self.null_patients.append(line.strip())

        # self.null_patients

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

                # verify string inside a list of string
                if patient_id not in self.null_patients:

                    eps = int("".join([c for c in info[1] if c.isdigit()]))
                    notes = pd.read_pickle(os.path.join(self.notes_dir, patient_id, "notes.pkl"))
                    notes[["CHARTTIME", "STORETIME", "CHARTDATE"]] = notes[["CHARTTIME", "STORETIME", "CHARTDATE"]].apply(pd.to_datetime)
                    # fill in the time, do two passes. Any not caught in the first pass will get helped by second
                    notes["CHARTTIME"] = notes["CHARTTIME"].fillna(notes["STORETIME"])
                    notes["CHARTTIME"] = notes["CHARTTIME"].fillna(value=notes["CHARTDATE"].map(lambda x: pd.Timestamp(x) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))

                    assert len(notes[notes["CHARTTIME"].isnull()]) == 0 # all of them should have been filled in.

                    # now, let's sort the notes
                    episode_specific_notes = notes[notes["EPISODES"] == eps].copy(deep=True)
                    if len(episode_specific_notes) > 0:
                        text_df = episode_specific_notes
                        text_df.sort_values("CHARTTIME", ascending=True, inplace=True)  # we want them sorted by increasing time

                        # unlike the other one, we found our performance acceptable. Therefore, we use only the first note.
                        text = text_df["TEXT"].iloc[0] #assuming sorted order

                        # join the texts together, or simply use the first one (according to starttime)
                        tokens = self.tokenizer.tokenize(text)

                        text_field = TextField(tokens, self.token_indexers)
                        label_field = LabelField(label)
                        fields = {'text': text_field, 'label': label_field}
                        yield Instance(fields)
                    else:
                        logger.warning("No text found for patient {}".format(patient_id))




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
    EMBED_DIMS = 300
    # turn the tokens into 300 dim embedding. Then, turn the embeddings into encodings
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=EMBED_DIMS, num_embeddings=vocab_size)})
    encoder = CnnEncoder(embedding_dim=EMBED_DIMS, ngram_filter_sizes = (2,3,4,5),
                         num_filters=5) # num_filters is a tad bit dangerous: the reason is that we have this many filters for EACH ngram f
    # encoder = BertPooler("bert-base-cased")
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