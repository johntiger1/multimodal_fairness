# the main purpose of the . notation is that it is _relative_, meaning the import will work
# regardless of the working directory that the main app is launched from
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary

import allennlp
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder


from DecompensationModel import DecompensationClassifier
from DecompensationReader import DecompensationReader
from typing import Dict, Iterable, List, Tuple

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
import torch
'''
Main file which will construct the DecompensationReader and DecompensationModel
and run them.

We will try it on a limited sample first, and then run it on a bigger sample. 

Also, we can rapidly try a seq2vec wrapper, LSTM 
'''

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# logger.debug("hello")

def build_dataset_reader(**kwargs) -> DatasetReader:
    return DecompensationReader(**kwargs, lazy=False)
#

# "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv"
'''
The issue is that this may for somereason read everything into memory
'''
def read_data(
    reader: DatasetReader,
        train_data_path: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
        valid_data_path: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv"
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read(train_data_path)
    validation_data = reader.read(valid_data_path)
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

'''
we can actually reuse the same model for each one. 
(depending on if different architectures work better or not)
'''
def build_model(vocab: Vocabulary,
                use_reg: bool = True) -> Model:
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

    #     construct the regularizer applicator
    regularizer_applicator = None
    if use_reg :
        l2_reg = L2Regularizer()
        regexes = [("embedder", l2_reg),
                   ("encoder", l2_reg),
                   ("classifier", l2_reg)
                   ]
        regularizer_applicator = RegularizerApplicator(regexes)

    return DecompensationClassifier(vocab, embedder, encoder,regularizer_applicator)
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
        num_epochs=50,
        optimizer=optimizer,
        cuda_device=0,
        validation_metric="+auc",
        patience=5

    )
    return trainer

'''
we pass in the vocab to ensure that we are speaking the same language!
'''
def run_training_loop_over_dataloaders(model,train_loader,dev_loader, args, use_gpu=False, batch_size =32,
                      serialization_dir = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments"):
    # move the model over, if necessary, and possible
    gpu_device = torch.device("cuda:0" if use_gpu  else "cpu")
    model = model.to(gpu_device)

    experiment_dir = os.path.join(serialization_dir, args.run_name)

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

    serialization_dir = experiment_dir

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this course, we need to do this.
    trainer = build_trainer(
        model,
        serialization_dir,
        train_loader,
        dev_loader
    )
    trainer.train()

    return model



def main():
    logger.setLevel(logging.CRITICAL)
    args = lambda x: None
    args.batch_size = 1024
    args.run_name = "30"
    args.train_data = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/train/listfile.csv"
    args.dev_data = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/test/listfile.csv"

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

    dataset_reader = build_dataset_reader(limit_examples=25000)

    dataset_reader.get_label_stats(args.train_data)
    for key in sorted(dataset_reader.stats.keys()):
        print("{} {}".format(key, dataset_reader.stats[key]))
    dataset_reader.get_label_stats(args.dev_data)

    for key in sorted(dataset_reader.stats.keys()):
        print("{} {}".format(key, dataset_reader.stats[key]))
    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader, args.train_data, args.dev_data)

    vocab = build_vocab(train_data + dev_data)

    # make sure to index the vocab before adding it
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    train_dataloader, dev_dataloader = build_data_loaders(train_data, dev_data)
    # del train_data
    # del dev_data

    # throw in all the regularizers to the regularizer applicators
    model = build_model(vocab, use_reg=False)
    model = run_training_loop_over_dataloaders(model, train_dataloader, dev_dataloader, args, use_gpu=True,
                                               batch_size=args.batch_size)

    logger.warning("We have finished training")


    results = evaluate(model, dev_dataloader, 0, None)

    print("we succ fulfilled it")
    with open(f"nice_srun_time_{args.run_name}.txt", "w") as file:
        file.write("it is done\n{}\nTook {}".format(results, time.time() - start_time))

    pass


if __name__ == "__main__":

    main()
    pass
