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


from MortalityModel import MortalityClassifier
from MortalityReader import MortalityReader
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

import matplotlib.pyplot as plt
from CONST import LOGGER_NAME

'''need this to get code to move over a dict'''
from allennlp.nn import util as nn_util


'''
Main file which will construct the DecompensationReader and DecompensationModel
and run them.

We will try it on a limited sample first, and then run it on a bigger sample. 

Also, we can rapidly try a seq2vec wrapper, LSTM 
'''

import logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)
# logger.debug("hello")

def build_dataset_reader(**kwargs) -> DatasetReader:
    return MortalityReader(**kwargs)
#

# "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv"
'''
The issue is that this may for some reason read everything into memory
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

    return MortalityClassifier(vocab, embedder, encoder,regularizer_applicator)
#
def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset, batch_size=64
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = DataLoader(train_data, batch_size=batch_size,)
    dev_loader = DataLoader(dev_data, batch_size=batch_size,)
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
def run_training_loop_over_dataloaders(model,train_loader,dev_loader, args):
    # move the model over, if necessary, and possible
    model = model.to(args.device )

    experiment_dir = args.serialization_dir

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

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
        args.serialization_dir,
        train_loader,
        dev_loader
    )
    trainer.train()

    return model


'''
On the given dataloader, we will make the predictions, and write them to file as well
'''
def make_predictions(model, eval_dataloader, args):

    for i,batch in enumerate(tqdm(eval_dataloader)):

        batch = nn_util.move_to_device(batch, args.device.index) #move it to the gpu

        output = model(**batch) #pass in the kwargs to the model, allowing it to process it
        # we can construct a dataframe, then serialize it as either a csv or directly pickle it

        # reconstruct a dictionary
        pandas_dict = {}
        for key in output:
            if key == "probs":
                pandas_dict[key] = output[key].detach().cpu()
            if key == "metadata":
                pandas_dict[key] = output[key]
        # otherwise, we can simply build the numpy array, and then assign the column names
                # for k in output["metadata"]: #metadata is actually a list! (of dict)
                #     pandas_dict["metadata"][k] = output["metadata"][k].cpu()
        # for key, value in ""
        # output = output.cpu() # move back to cpu first
        # we should be able to just tie everything together
        predictions = pd.DataFrame.from_dict(pandas_dict)

        predictions.to_csv(os.path.join(args.serialization_dir, f"predictions_{i}.csv"))
        # write the predictions to csv

    # labels and probs will be gotten, along with the metadata
    pass

def main():
    logger.setLevel(logging.CRITICAL)
    args = lambda x: None
    args.batch_size = 1024
    args.run_name = "31"
    args.train_data = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv"
    args.dev_data = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv"
    args.serialization_dir = os.path.join("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments",args.run_name)
    args.use_gpu = True
    args.device = torch.device("cuda:0" if args.use_gpu  else "cpu")
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

    dataset_reader = build_dataset_reader(limit_examples=100)

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
    model = run_training_loop_over_dataloaders(model, train_dataloader, dev_dataloader, args)

    logger.warning("We have finished training")


    results = evaluate(model, dev_dataloader, 0, None)
    make_predictions(model, dev_dataloader, args)
    print("we succ fulfilled it")
    with open(f"nice_srun_time_{args.run_name}.txt", "w") as file:
        file.write("it is done\n{}\nTook {}".format(results, time.time() - start_time))

    pass

def get_preprocessed_stats(train_data_path="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/train/listfile.csv",
                           dev_data_path="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/test/listfile.csv"):
    logger.setLevel(logging.CRITICAL)
    print(f"in main , the logger is {logger} and we have {__name__}")

    dataset_reader = build_dataset_reader(limit_examples = 2500)
    args = lambda x: None
    args.batch_size = 1024
    args.run_name = "31"
    args.train_data = train_data_path
    args.dev_data = dev_data_path

    train_note_stats = dataset_reader.get_note_stats(args.train_data, "train")
    test_note_stats = dataset_reader.get_note_stats(args.dev_data, "test")
    # merge the dictionaries
    all_note_stats = {**train_note_stats, **test_note_stats}

    all_note_lengths = []
    for lst in all_note_stats.values():
        all_note_lengths = all_note_lengths + lst
    fig, ax = plt.subplots()
    ax.hist(all_note_lengths, range=(0, 25000), bins=100, rwidth=0.9)
    ax.set_title("histogram of note lengths for decompensation")
    fig.savefig(f"decomp_note_length_hist.png")
    # this can also be done much more easily from pandas/jupyter.
    # therefore, we do not do it too in-depth here




if __name__ == "__main__":
    # get_preprocessed_stats()
    main()
    pass
