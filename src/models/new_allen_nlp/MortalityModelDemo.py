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

# import the regularization
from allennlp.nn.regularizers import L2Regularizer, RegularizerApplicator

import pandas as pd
import os
import gc
from tqdm.auto import tqdm
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
                 max_tokens: int = 768,
                 listfile: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
                 notes_dir: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes",
                 skip_patients_file: str ="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/null_patients.txt",
                 stats_write_dir: str="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/",
                 all_stays: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/root/all_stays.csv"

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
        self.stats_write_dir = stats_write_dir
        self.all_stays_path = all_stays
        self.all_stays_df = self.get_all_stays()
        # self.null_patients

    def get_label_stats(self, file_path: str):
        '''
        Gets label (mortality) stats
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

    '''
    Gets stats for the data listed at the datapath
    '''
    def get_note_stats(self, file_path):
        self.note_stats = {}
        exclusions = 0
        with open(file_path, "r") as file, \
                open(os.path.join(self.stats_write_dir, "note_lengths.txt") , "a") as note_length_file:
            file.readline() # could also pandas readcsv and ignore first line
            for line in tqdm(file):
                info_filename, label = line.split(",")
                info = info_filename.split("_")
                patient_id = info[0]

                # verify string inside a list of string
                if patient_id not in self.null_patients: # could also just do try except here

                    eps = int("".join([c for c in info[1] if c.isdigit()]))
                    notes = pd.read_pickle(os.path.join(self.notes_dir, patient_id, "notes.pkl"))
                    notes[["CHARTTIME", "STORETIME", "CHARTDATE"]] = notes[["CHARTTIME", "STORETIME", "CHARTDATE"]].apply(pd.to_datetime)
                    # fill in the time, do two passes. Any not caught in the first pass will get helped by second
                    notes["CHARTTIME"] = notes["CHARTTIME"].fillna(notes["STORETIME"])
                    notes["CHARTTIME"] = notes["CHARTTIME"].fillna(value=notes["CHARTDATE"].map(lambda x: pd.Timestamp(x) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))

                    assert len(notes[notes["CHARTTIME"].isnull()]) == 0 # all of them should have been filled in.

                    # now, let's sort the notes
                    episode_specific_notes = notes[notes["EPISODES"] == eps].copy(deep=True)

                    # hadm_id = episode_specific_notes.groupby(["HADM_ID"]).agg({""}) # hadm ids seem to 1-to1 correspond to episodes
                    hadm_id = episode_specific_notes["HADM_ID"]
                    one_hadm_id = hadm_id.unique()
                    logger.info(type(one_hadm_id))
                    assert (one_hadm_id.shape[0]) == 1
                    assert len(one_hadm_id) == 1

                    icu_intime = self.all_stays_df[ self.all_stays_df["HADM_ID"] == one_hadm_id[0]]

                    # we are assuming that the intime is not null
                    intime_date = pd.Timestamp(icu_intime["INTIME"].iloc[0]) # iloc will automatically extract once you get to the base element
                    logger.info(type(intime_date))
                    intime_date_plus_two_days = pd.Timestamp(intime_date) + pd.Timedelta(days=2)

                    # all notes up to two days. Including potentially previous events.
                    mask = ( episode_specific_notes["CHARTTIME"] > intime_date) & (episode_specific_notes["CHARTTIME"] <= intime_date_plus_two_days)
                    all_mask = (episode_specific_notes["CHARTTIME"] <= intime_date_plus_two_days)

                    time_episode_specific_notes = episode_specific_notes[mask].copy(deep=True)

                    logger.debug("Went from {} to {} notes\n".format(len(episode_specific_notes), len(time_episode_specific_notes)))
                    # filter all of them, based on the all stays info
                    # do the episode and patient, associate to a specific hadm? i.e. are there 1 to 1 mappings here


                    # first, get the icu intime
                    # then, get the
                    # icu_intime = self.all_stays_df[]

                    # with open(os.path.join(self.stats_write_dir, "num_notes.txt"), "a") as notes_dir:

                    if len(time_episode_specific_notes) > 0:

                        text_df = time_episode_specific_notes
                        text_df.sort_values("CHARTTIME", ascending=True, inplace=True)  # we want them sorted by increasing time

                        # unlike the other one, we found our performance acceptable. Therefore, we use only the first note.
                        text = text_df["TEXT"].iloc[0] #assuming sorted order
                        tokens = self.tokenizer.tokenize(text)
                        if patient_id in self.note_stats:
                            logger.warning("Encountering the patient another time, for another episode {} {}".format(patient_id, eps))
                        self.note_stats[patient_id] = len(tokens)
                        if int(patient_id)%1000==0:
                            logger.info("text for patient {} \n: {}".format(patient_id,text))
                            logger.info("end of text for patient {} \n".format(patient_id))

                        xs = ""
                        if len(time_episode_specific_notes) > 1:

                        # lets try to join both of them
                            xs = text_df["TEXT"].iloc[1] #assuming sorted order
                        else:
                            logger.info("pat, eps: {} {} had only one note".format(patient_id, eps))

                    else:
                        logger.warning("No text found for patient {}. This is with the 48 hour\n. ".format(patient_id))
                        exclusions +=1
            sorted_dict = sorted(self.note_stats.items(), key=lambda tup: tup[1])
            note_length_file.write("For this file {}\n".format(file_path))
            for tup in sorted_dict:
                note_length_file.write("{} {}\n".format(tup[1], tup[0]))

        logger.critical("With 48 hour windowing, removed {}\n".format(exclusions))
        return self.note_stats

    def get_all_stays(self):
        my_stays_df = pd.read_csv(self.all_stays_path)
        return my_stays_df


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
                if patient_id not in self.null_patients: # could also just do try except here

                    eps = int("".join([c for c in info[1] if c.isdigit()]))
                    notes = pd.read_pickle(os.path.join(self.notes_dir, patient_id, "notes.pkl"))
                    notes[["CHARTTIME", "STORETIME", "CHARTDATE"]] = notes[["CHARTTIME", "STORETIME", "CHARTDATE"]].apply(pd.to_datetime)
                    # fill in the time, do two passes. Any not caught in the first pass will get helped by second
                    notes["CHARTTIME"] = notes["CHARTTIME"].fillna(notes["STORETIME"])
                    notes["CHARTTIME"] = notes["CHARTTIME"].fillna(value=notes["CHARTDATE"].map(lambda x: pd.Timestamp(x) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))

                    assert len(notes[notes["CHARTTIME"].isnull()]) == 0 # all of them should have been filled in.

                    # now, let's sort the notes
                    episode_specific_notes = notes[notes["EPISODES"] == eps]

                    hadm_id = episode_specific_notes["HADM_ID"]
                    one_hadm_id = hadm_id.unique()


                    icu_intime = self.all_stays_df[self.all_stays_df["HADM_ID"] == one_hadm_id[0]]

                    # we are assuming that the intime is not null
                    intime_date = pd.Timestamp(icu_intime["INTIME"].iloc[
                                                   0])  # iloc will automatically extract once you get to the base element
                    intime_date_plus_two_days = pd.Timestamp(intime_date) + pd.Timedelta(days=2)

                    # all notes up to two days. Including potentially previous events.
                    mask = (episode_specific_notes["CHARTTIME"] > intime_date) & (
                                episode_specific_notes["CHARTTIME"] <= intime_date_plus_two_days)

                    time_episode_specific_notes = episode_specific_notes[mask].copy(deep=True)
                    # with open(os.path.join(self.stats_write_dir, "num_notes.txt"), "a") as notes_dir:

                    if len(time_episode_specific_notes) > 0:

                        text_df = time_episode_specific_notes
                        text_df.sort_values("CHARTTIME", ascending=False, inplace=True)  # we want them sorted by increasing time

                        # unlike the other one, we found our performance acceptable. Therefore, we use only the first note.
                        text = " ".join(text_df["TEXT"].tolist())
                        #
                        # iloc[0] #assuming sorted order
                        #
                        # xs = ""
                        # if len(time_episode_specific_notes) > 1:
                        #
                        # # lets try to join both of them
                        #     xs = text_df["TEXT"].iloc[1] #assuming sorted order
                        # else:
                        #     logger.info("pat, eps: {} {} had only one note".format(patient_id, eps))
                        # text = xs + " " + text

                        # join the texts together, or simply use the first one (according to starttime)
                        tokens = self.tokenizer.tokenize(text)

                        text_field = TextField(tokens, self.token_indexers)
                        label_field = LabelField(label)
                        fields = {'text': text_field, 'label': label_field}
                        yield Instance(fields)
                    else:
                        logger.warning("No text found for patient {}".format(patient_id))
                        # in this case, we ignore the patient



'''
We can also look at dropout and other techniques which we do here!
'''
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
        # reg_loss = self.get_regularization_penalty() # should not have to manually apply the regularization
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        self.accuracy(logits, label)
        preds = logits.argmax(-1)
        self.auc(preds, label)
        output = {'loss': loss, 'probs': probs}
        return output

    '''this is called'''
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                "auc":self.auc.get_metric(reset)}
#
#
def build_dataset_reader(**kwargs) -> DatasetReader:
    return MortalityReader(**kwargs, lazy=False)
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
        num_epochs=25,
        optimizer=optimizer,
        cuda_device=0
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

'''
we pass in the vocab to ensure that we are speaking the same language!
'''
def run_training_loop(model, dataset_reader, vocab, args, use_gpu=False, batch_size =32,
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
    gc.collect()

    train_loader, dev_loader = build_data_loaders_from_reader(dataset_reader,vocab, batch_size=batch_size)
    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this course, we need to do this.
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
    logger.setLevel(logging.CRITICAL)
    args = lambda x: None
    args.batch_size = 32
    args.run_name = "9"
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

    dataset_reader.get_label_stats("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv")
    for key in sorted(dataset_reader.stats.keys()):
        print("{} {}".format(key,dataset_reader.stats[key]))
    dataset_reader.get_label_stats("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv")

    for key in sorted(dataset_reader.stats.keys()):
        print("{} {}".format(key,dataset_reader.stats[key]))
    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)

    # make sure to index the vocab before adding it
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    train_dataloader, dev_dataloader = build_data_loaders(train_data, dev_data)
    # del train_data
    # del dev_data

    # throw in all the regularizers to the regularizer applicators
    model = build_model(vocab,use_reg=False)
    model = run_training_loop_over_dataloaders(model, train_dataloader, dev_dataloader, args,use_gpu=True, batch_size=args.batch_size)

    logger.warning("We have finished training")
    # Now we can evaluate the model on a new dataset.
    test_data = dataset_reader.read(
        '/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv')
    test_data.index_with(model.vocab)
    data_loader = DataLoader(test_data, batch_size=args.batch_size)

    # results = evaluate(model, data_loader, -1, None)
    # print(results)

    results = evaluate(model, data_loader, 0, None)

    print("we succ fulfilled it")
    with open("nice_srun_time.txt", "w") as file:
        file.write("it is done\n{}\nTook {}".format(results, time.time() - start_time))


'''
get the stats of the preprocessed notes
'''
def get_preprocessed_stats():
    logger.setLevel(logging.CRITICAL)
    dataset_reader = build_dataset_reader()
    train_note_stats = dataset_reader.get_note_stats("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv")
    test_note_stats = dataset_reader.get_note_stats("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv")
    # merge the dictionaries
    all_note_stats = {**train_note_stats, **test_note_stats}
    sorted_dict = sorted(all_note_stats.items(), key=lambda tup: tup[1])
    with open("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/note_lengths.txt", "a") as note_length_file:
        for tup in sorted_dict:
            note_length_file.write("{} {}\n".format(tup[1], tup[0]))


if __name__ == __name__:
    main()
    # get_preprocessed_stats()
    # dataset_reader = build_dataset_reader(all_stays="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/root/all_stays.csv")
    # stays_df = dataset_reader.get_all_stays()
    # print(len(stays_df))
