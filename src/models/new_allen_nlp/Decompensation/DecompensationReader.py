
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
get the logger, if it is available
'''
import logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("hello")

@DatasetReader.register("DecompensationReader")
class DecompensationReader(DatasetReader):
    def __init__(self,
                 lazy: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = 768*4,
                 listfile: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
                 notes_dir: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes",
                 skip_patients_file: str ="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/null_patients.txt",
                 stats_write_dir: str="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/",
                 all_stays: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/root/all_stays.csv",
                 limit_examples: int = None,
                 use_preprocessing: bool = False

    ):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.listfile = listfile
        self.notes_dir = notes_dir
        logger.critical(f"we are getting the max tokens {self.max_tokens}")
        self.null_patients = []
        with open(skip_patients_file, "r") as file:
            for line in file:
                self.null_patients.append(line.strip())
        self.stats_write_dir = stats_write_dir
        self.all_stays_path = all_stays
        self.all_stays_df = self.get_all_stays()
        self.use_preprocessing = use_preprocessing
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
                info_filename, _, label = line.split(",")
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
                info_filename, time, label = line.split(",")
                time = float(time)
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
                    intime_date_plus_time = pd.Timestamp(intime_date) + pd.Timedelta(hours=int(time))

                    # all notes up to two days. Including potentially previous events.
                    mask = ( episode_specific_notes["CHARTTIME"] > intime_date) & (episode_specific_notes["CHARTTIME"] <= intime_date_plus_time)
                    all_mask = (episode_specific_notes["CHARTTIME"] <= intime_date_plus_time)

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

        logger.critical("With decompensation windowing, removed {}\n".format(exclusions))
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
                cur_tokens = 0
                info_filename, time, label = line.split(",")
                time = float(time)
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
                    intime_date_plus_time = pd.Timestamp(intime_date) + pd.Timedelta(hours=int(time))

                    # all notes up to two days. Including potentially previous events.
                    mask = (episode_specific_notes["CHARTTIME"] > intime_date) & (
                                episode_specific_notes["CHARTTIME"] <= intime_date_plus_time)

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
                        # tokens = self.tokenizer.tokenize(text)[:self.max_tokens]
                        if self.use_preprocessing:
                            token_sent_stream = preprocess_mimic(text)
                            tokens = []
                            cur_tokens = 0
                            for i,token_sent in enumerate(token_sent_stream):
                                if cur_tokens > self.max_tokens: break
                                cur_tokens += len(token_sent.split())
                                tokens.append(token_sent)

                            text = " ".join(tokens) #overwrite the text!
                        tokens = self.tokenizer.tokenize(text)[:self.max_tokens]

                        text_field = TextField(tokens, self.token_indexers)
                        label_field = LabelField(label)
                        fields = {'text': text_field, 'label': label_field}
                        yield Instance(fields)
                    else:
                        logger.warning("No text found for patient {}".format(patient_id))
                        # in this case, we ignore the patient

