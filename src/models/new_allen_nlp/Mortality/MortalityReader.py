
import tempfile
from typing import Dict, Iterable, List, Tuple
from overrides import overrides

import torch

import allennlp
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField, MetadataField, MultiLabelField
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
import torch
import matplotlib.pyplot as plt
from CONST import LOGGER_NAME
'''
get the logger, if it is available
'''
import logging
import numpy as np
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)
logger.debug("hello")

@DatasetReader.register("MortalityReader")
class MortalityReader(DatasetReader):
    def __init__(self,
                 lazy: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = 768*4,
                 train_listfile: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
                 test_listfile: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv",
                 notes_dir: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes",
                 skip_patients_file: str ="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/null_patients.txt",
                 stats_write_dir: str="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/",
                 all_stays: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/root/all_stays.csv",
                 limit_examples: int = None,
                 use_preprocessing: bool = False,
                 num_classes: int=2,
                 mode: str='train',
                 data_type: str="MORTALITY",
                 args=None,
                 hadm2eps_path: str="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/hadm2episode.dict"


    ):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.train_listfile = train_listfile
        self.test_listfile = test_listfile
        self.notes_dir = notes_dir
        self.use_preprocessing = use_preprocessing

        logger.critical(f"we are getting the max tokens {self.max_tokens} "
                        f"and use_preproc is {self.use_preprocessing}")
        self.null_patients = []
        with open(skip_patients_file, "r") as file:
            for line in file:
                self.null_patients.append(line.strip())
        self.stats_write_dir = stats_write_dir
        self.all_stays_path = all_stays
        self.all_stays_df = self.get_all_stays()
        self.limit_examples = limit_examples
        self.cur_examples = 0
        self.lengths = []
        self.num_classes = num_classes
        self.mode =  mode
        self.sampled_idx = {}
        self.data_type = data_type
        self.args = args

        self.get_idx() #realistically, only the train_idx will be set, and we simply need to compare against
        # self.null_patients
        self.vocab = None
        self.hadm2eps_path = hadm2eps_path
        self.listfile_df = pd.read_csv(train_listfile)

        if self.data_type == "PHENOTYPING" or self.data_type == "DECOMPENSATION":
            self.labels_start_idx = 2
        elif self.data_type == "MORTALITY":
            self.labels_start_idx = 1

        self.labels = list(self.listfile_df.columns[self.labels_start_idx:])
    # def set_mode(self, mode: str):
    #     if mode == "train":
    #         self.limit_examples = None
    #     else:
    #
    #     pass

    def get_idx(self):

        train_sampler = self.get_sampler(self.train_listfile)
        self.sampled_idx["train"] = list(train_sampler)
        self.train_sampler = train_sampler

        test_sampler = self.get_sampler(self.test_listfile)
        self.sampled_idx["valid"] = list(test_sampler)
        self.test_sampler = test_sampler

        if self.limit_examples:
            self.sampled_idx["train"] = self.sampled_idx["train"][:self.limit_examples]
            self.sampled_idx["valid"] = self.sampled_idx["valid"][:self.limit_examples]

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
    Parses the line, according to the mode. Returns a dict with the proper keys set
    Could also have implemented this with a DF instead
    '''
    def parse_line(self, line):
        info_dict = {}
        mapping_dict = {}

        if self.data_type == "MORTALITY":
            headers = ["filename", "label"]
        elif self.data_type == "DECOMPENSATION":
            headers = ["filename", "time", "label"]
        elif self.data_type == "PHENOTYPING":
            headers = ["filename", "time", "label"]
        else:
            headers = ["filename", "time", "label"]


        for i,header in enumerate(headers):
            mapping_dict[header] = i #can also use a dict comprehension here

        info_array = line.split(",")
        for key in mapping_dict:
            if key == "label":
                info_dict[key] = int(info_array[mapping_dict[key]])
            elif key == "time":
                info_dict[key] = float(info_array[mapping_dict[key]])
            else:
                info_dict[key] = info_array[mapping_dict[key]]

        return info_dict


    '''
    Reads in all the labels, and forms a sampler, according to a balanced approach. 
    '''

    def get_sampler(self, listfile: str = ""):
        self.labels = []

        # read out the personal statement, and expand upon this
        # current events
        # politics and area studies
        # controversial issues: pipeline protests
        # saudi arms deal!
        # conservative, etc.
        # african and canadian politics
        # excerpt from the referee letter!
        # sampling_num_classes
        sampling_num_classes = None
        if self.data_type == "DECOMPENSATION" or self.data_type == "MORTALITY":
            sampling_num_classes = 2
        else:
            sampling_num_classes = 25



        self.class_counts = np.zeros(sampling_num_classes) # fix sampling for phenotypes
        with open(listfile, "r") as file:
            file.readline()
            for line in file:
                info_dict = self.parse_line(line)

                self.labels.append([info_dict["label"]])
                self.class_counts[int(info_dict["label"])] += 1

        # now, we assign the weights to ALL the class labels
        self.class_weights = 1/self.class_counts
        # essentially, assign the weights as the ratios, from the self.stats stuff

        all_label_weights = self.class_weights[self.labels].squeeze() #produce an array of size labels, but looking up the value in class weights each time
        num_samples = self.limit_examples if self.limit_examples else len(all_label_weights)
        num_samples = min(num_samples, len(all_label_weights))

        if self.args.sampler_type == "balanced":
            sampler  = torch.utils.data.sampler.WeightedRandomSampler(weights=all_label_weights,
                                                                      num_samples=num_samples,
                                   replacement = False)
        elif self.args.sampler_type == "random":
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=[i for i in range(len(all_label_weights))])
            # sampler = list(sampler)[:num_samples]
        else:
            logger.critical("Weird sampler specified \n")
            sampler = None
        return sampler

    def get_sampler_from_dataset(self, dataset):
        self.labels = []
        self.class_counts = np.zeros(2)

        for data in dataset: # could replace this with an arbitrary data source, and we just yield from it
            info_dict = data.fields
            label = int(info_dict["label"].label)
            self.labels.append(label)
            self.class_counts[label] += 1

        # now, we assign the weights to ALL the class labels
        self.class_weights = 1/self.class_counts
        # essentially, assign the weights as the ratios, from the self.stats stuff

        all_label_weights = self.class_weights[self.labels] #produce an array of size labels, but looking up the value in class weights each time
        num_samples = self.limit_examples if self.limit_examples else len(all_label_weights)
        num_samples = min(num_samples, len(all_label_weights))

        if self.args.sampler_type == "balanced":
            sampler  = torch.utils.data.sampler.WeightedRandomSampler(weights=all_label_weights,
                                                                      num_samples=num_samples,
                                                                      replacement = False)
        else:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=[i for i in range(len(all_label_weights))])
            # sampler = list(sampler)[:num_samples]

        return sampler #now that we have a sampler, we can do things: pass it into the dataloader

    '''
    Creates and saves a histogram of the note lengths
    '''
    def make_lengths_histogram(self):
        pass

    '''
    Gets stats for the data listed at the datapath
    '''
    def get_note_stats(self, file_path, name="train"):
        print(f"in note stats, the logger is {logger} and we have {__name__}")
        print(logger.getEffectiveLevel())
        from collections import defaultdict
        self.note_stats = defaultdict(list)
        exclusions = 0

        num_examples = 0

        with open(file_path, "r") as file:
            for line in file:
                num_examples+=1
        with open(file_path, "r") as file, \
                open(os.path.join(self.stats_write_dir, "note_lengths.txt") , "a") as note_length_file:
            file.readline() # could also pandas readcsv and ignore first line
            for example_number,line in enumerate(tqdm(file, total=num_examples)):

                if self.mode != "test" and self.limit_examples and example_number > self.limit_examples:
                    break
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
                    intime_date_plus_time = pd.Timestamp(intime_date) + pd.Timedelta(days=2)

                    # all notes up to two days. Including potentially previous events.
                    mask = ( episode_specific_notes["CHARTTIME"] > intime_date) & (episode_specific_notes["CHARTTIME"] <= intime_date_plus_time)
                    all_mask = (episode_specific_notes["CHARTTIME"] <= intime_date_plus_time)

                    time_episode_specific_notes = episode_specific_notes[mask].copy(deep=True)

                    logger.debug("Went from {} to {} notes\n".format(len(episode_specific_notes), len(time_episode_specific_notes)))

                    if len(time_episode_specific_notes) > 0:

                        text_df = time_episode_specific_notes
                        text_df.sort_values("CHARTTIME", ascending=True, inplace=True)  # we want them sorted by increasing time

                        # unlike the other one, we found our performance acceptable. Therefore, we use only the first note.
                        text = " ".join(text_df["TEXT"].tolist()) #assuming sorted order

                        tokens = self.tokenizer.tokenize(text)
                        if patient_id in self.note_stats:
                            logger.info("Encountering the patient another time, for another episode {} {}".format(patient_id, eps))
                        self.note_stats[patient_id].append(len(tokens) )# the same patient id can be encoutnered for multiple episodes


                        if int(patient_id)%1000==0:
                            logger.info("text for patient {} \n: {}".format(patient_id,text))
                            logger.info("end of text for patient {} \n".format(patient_id))


                    else:
                        logger.warning("No text found for patient {}. This is with the time hour {} window\n. ".format(patient_id, 48))
                        exclusions +=1

            '''below code is functionally useless; much better to visualize with plot'''
        fig, ax = plt.subplots()
        # let's flatten a dictionary of lists.
        note_lengths = []
        for lst in self.note_stats.values():
            note_lengths.extend(lst)
        ax.hist(note_lengths, range=(0, max(note_lengths)), bins=100, rwidth=0.9 )
        ax.set_title("Histogram of total note lengths")

        fig.savefig(os.path.join(self.stats_write_dir, f"{name}_decomp_note_length_hist.png"))

        logger.critical("For {} With decompensation windowing, removed {}\n".format(name, exclusions))
        return self.note_stats

    def get_all_stays(self):
        my_stays_df = pd.read_csv(self.all_stays_path)
        return my_stays_df

    @overrides
    def _read(self, listfile_path: str) -> Iterable[Instance]:
        '''NOTE: because this is an overrides, it CANNOT accept another arg!'''
        '''This function is only expected to be called with lazy set to FALSE. '''
        '''Expect: one instance per line'''
        logger.critical("read method is called")

        if self.mode != "test":
            sampled_idx = self.sampled_idx[self.mode]

        listfile_df = pd.read_csv(listfile_path)

        def str_build(row):
            labels = []
            for col in row.index:
                if row[col] == 1:
                    labels.append(col)
            # labels.join()
            return labels

        for idx,row in listfile_df.iterrows():

            if self.mode == "test" or idx  in sampled_idx : #when test, use everything
                list_labels = str_build(row.iloc[self.labels_start_idx:])
                multi_labels = MultiLabelField(list_labels)
                time = row.get("period_length", default=48)  # float(48) #hardcode to 48
                info = row.get("stay").split("_")
                label = row.get("y_true", -1)
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




                    if self.data_type != "PHENOTYPING":
                        icu_intime = self.all_stays_df[self.all_stays_df["HADM_ID"] == one_hadm_id[0]]

                        # we are assuming that the intime is not null
                        intime_date = pd.Timestamp(icu_intime["INTIME"].iloc[
                                                       0])  # iloc will automatically extract once you get to the base element
                        intime_date_plus_time = pd.Timestamp(intime_date) + pd.Timedelta(hours=int(time))

                        # all notes up to two days. Including potentially previous events.
                        mask = (episode_specific_notes["CHARTTIME"] > intime_date) & (
                                    episode_specific_notes["CHARTTIME"] <= intime_date_plus_time)

                        time_episode_specific_notes = episode_specific_notes[mask].copy(deep=True)

                    else:
                        time_episode_specific_notes = episode_specific_notes.copy(deep=True)

                    if len(time_episode_specific_notes) > 0:

                        text_df = time_episode_specific_notes
                        text_df.sort_values("CHARTTIME", ascending=False, inplace=True)  # we want them sorted by increasing time


                        # unlike the other one, we found our performance acceptable. Therefore, we use only the first note.
                        text = " ".join(text_df["TEXT"].tolist())


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
                        label_field = LabelField(str(label).strip())
                        meta_data_field = MetadataField({"patient_id": patient_id,
                                                         "episode": eps,
                                                         "hadm_id": one_hadm_id[0], # just the specific value
                                                         "time" : time #yield the time too
                                                         })
                        fields = {'text': text_field, 'label': multi_labels, "metadata": meta_data_field}

                        yield Instance(fields)

                        # after the generator yields, code will return here. (think of yield as a pause)
                        # self.cur_examples += 1

                    else:
                        logger.warning("No text found for patient {}".format(patient_id))
                        # in this case, we ignore the patient

            else:
                logger.debug("we are skipping  some indices {}".format(idx))
