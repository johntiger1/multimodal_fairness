import sys
sys.path.insert(1, "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness")
from mimic3benchmark.readers import DecompensationReader, InHospitalMortalityReader
import pandas as pd
import logging
import glob
import os
import pandas as pd
import pickle
from tqdm.auto import  tqdm
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("hello")
# logger.setLevel(level=logging.INFO)

'''
logging info: Debug is roughly the lowest level

'''
'''
This script will extract all the notes for the patients.

Assuming you have the patient vital data set up according to the benchmark,
then this script will ensure that the relevant notes are dumped into the same folder
folder, as a pandas df.  

Further post-processing may be possible! (for instance, to extract causally masked
notes). 
'''


'''
Extracts notes, going over the TOTAL dataset.
It also generates a mapping: {patient: hadm_id}
and {hadm_id: eps}.

FOR an episode, it corresponds to potentially a series of hadm_ids 

Therefore, we want to ensure 1:1 correspondence with hadm_id to eps 

'''
def extract_notes(data_path="data/root", output_dir ="data/extracted_notes"):
    '''

    '''
    stats = {}
    import numpy as np

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    #TODO: simply only take in the hadm ids, found in stays.csv!
    notes_table = pd.read_csv("data/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv")
    hadm_notes = notes_table[notes_table["HADM_ID"].notna()]
    logger.warning("Dropping rows where HADM_ID is null: {} {}".format(len(notes_table), len(hadm_notes)))
    del notes_table

    total = 0

    patient2hadm = defaultdict(list)     # "patient" -> ["hadm1", "hadm2"...]
    hadm2episode = {} # hadm1 -> eps1

    for root, dir, file in os.walk(data_path):
        total+=1
        splits = {"train": 29000, "test": 5000}
        curr_dir = root.split(os.path.sep)[-1]
        if curr_dir in splits:
            for split in tqdm(dir, total=splits[curr_dir]):

                patient_id = int(split)
                patient_hadm2episode_mapping = {}
                with open(os.path.join(root, split, "stays.csv")) as stays_file:
                    for idx, line in enumerate(stays_file):
                        if idx > 0:
                            hadm_id = int(line.split(",")[1])
                            patient2hadm[patient_id].append(hadm_id)
                            if hadm_id in hadm2episode:
                                logger.error("oops")
                            hadm2episode[hadm_id] = idx
                            patient_hadm2episode_mapping[hadm_id] = idx

                patient_notes_idx = (hadm_notes["SUBJECT_ID"] == patient_id)
                patient_notes = hadm_notes[ patient_notes_idx]


                patient_notes["EPISODES"] = patient_notes["HADM_ID"].map(patient_hadm2episode_mapping)

                no_eps_mapping = patient_notes["EPISODES"].isnull().sum()
                if no_eps_mapping > 0:
                    logger.info("{} notes that don't correspond to episode encountered in patient {}".format(
                        no_eps_mapping, patient_id))


                output_subdir = os.path.join(output_dir, str(patient_id))
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir, exist_ok=True)

                output_location = os.path.join(output_subdir, "notes.pkl")
                patient_notes.to_pickle(output_location)
            #     print (output_location)
            #     break
            # break

    # save the hadm2episode index. Note that, it is also useful to have the specific patient as well

    pickle.dump(patient2hadm, open(os.path.join(output_dir, "patient2hadm.dict"), "wb"))
    pickle.dump(hadm2episode, open(os.path.join(output_dir, "hadm2episode.dict"), "wb"))

    logging.info("{} entries processed".format(total))


def test_merge(notes_path = "data/extracted_notes", vitals_path = "data/in-hospital-mortality",
               hadm2index_path =""
               ):
    '''
    Tests that we can load and merge everything
    Note that os.walk will "visit" paths and nodes repeatedly
    For instance, if we have a sub-directory, then it will be traversed twice:
    Once as a subdirectory, and then once as a main directory

    In particular, it will always split it such that there is a dir, subdir and file
    Root must be given, otherwise the others can be empty
    they may be empty, but the format must be maintained

    We cannot walk the directories directly. That is, for each of the directories, we would like to
    immediately, get the list of all their fles, but this is not possible!

    SINCE it is not a 3 way! It is just 2 levels. We need the dirs AND the files!
    i.e. the files in the same level
    '''
    for dir, subdir, file in os.walk(vitals_path):
        logger.debug("{},{},{}".format(dir,subdir,file))
        pardir_name = dir.split(os.path.sep)[-1]
        logger.info(pardir_name)
        # if pardir_name== "train":
        #     for fil in file:
        #         print(fil)

        # if subdir.isdigit():
        #     pass

        pass

    pass

def get_mappings(output_dir="data/extracted_notes"):
    pickle.load(os.path.join(output_dir, "patient2hadm.dict"))
    return

'''
Loads a mapping. 
'''
def load_map():
    output_dir = "data/extracted_notes"
    with open(os.path.join(output_dir, "patient2hadm.dict")) as file:
        p2hadm = pd.read_pickle(os.path.join(output_dir, "patient2hadm.dict"))
    print(p2hadm)
    pass

'''

Build mapping dictionaries. 

patient2hadm: {patient:hadm}  
hadm2eps:  {hadm: eps}

(we verified the veracity of this)
Now, we just need to do the following: get all the notes corresponding to an episode, 
or equivalently to an hadm.
 
Additionally, join all the notes corresponding to a mortality event (up to 512 tokens).
 
'''
def build_mapping_dicts(data_path="data/root", output_dir ="data/extracted_notes"):
    total = 0
    patient2hadm = defaultdict(list)     # "patient" -> ["hadm1", "hadm2"...]
    hadm2episode = {} # hadm1 -> eps1
    for root, dir, file in os.walk(data_path):
        total+=1
        splits = {"train": 28728, "test": 5000}
        curr_dir = root.split(os.path.sep)[-1]
        if curr_dir in splits:
            for split in tqdm(dir, total=splits[curr_dir]):
                patient_id = int(split)
                patient_hadm2episode_mapping = {}
                with open(os.path.join(root, split, "stays.csv")) as stays_file:
                    for idx, line in enumerate(stays_file):
                        if idx > 0:
                            hadm_id = int(line.split(",")[1])
                            patient2hadm[patient_id].append(hadm_id)
                            if hadm_id not in hadm2episode:
                                logger.error("oops")
                            hadm2episode[hadm_id] = idx
                            patient_hadm2episode_mapping[hadm_id] = idx



if __name__ == "__main__":
    # extract_notes()
    build_mapping_dicts()
    # test_merge()
    # df = pd.read_pickle("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes/1819/notes.pkl")
    # print(df)


