from mimic3benchmark.readers import DecompensationReader, InHospitalMortalityReader
import pandas as pd
import logging
import glob
import os
import pandas as pd

from tqdm.auto import  tqdm
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("hello")
'''
This script will extract all the notes for the patients.

Assuming you have the patient vital data set up according to the benchmark,
then this script will ensure that the relevant notes are dumped into the same folder
folder, as a pandas df.  

Further post-processing may be possible! (for instance, to extract causally masked
notes). 
'''


'''
Extracts notes, going over the TOTAL dataset
'''
def extract_notes(data_path="data/root", output_dir ="data/extracted_notes"):

    '''
    load a copy of the notes CSV for processing

    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    notes_table = pd.read_csv("data/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv")
    total = 0
    for root, dir, file in os.walk(data_path):
        total+=1
        splits = {"train": 35000, "test": 5000}
        curr_dir = root.split(os.path.sep)[-1]
        if curr_dir in splits:
            for split in tqdm(dir, total=splits[curr_dir]):

                patient_id = int(split)
                patient_hadm2episode_mapping = {}
                with open(os.path.join(root, split, "stays.csv")) as stays_file:
                    for idx, line in enumerate(stays_file):
                        if idx > 0:
                            hadm_id = int(line.split(",")[1])
                            patient_hadm2episode_mapping[hadm_id] = idx

                patient_notes_idx = (notes_table["SUBJECT_ID"] == patient_id)
                patient_notes = notes_table[ patient_notes_idx]

                patient_notes["EPISODES"] = patient_notes["HADM_ID"].map(patient_hadm2episode_mapping)

                logger.error("{} null values encountered in patient {}".format(
                    patient_notes["EPISODES"].isnull().sum(), patient_id))


                output_subdir = os.path.join(output_dir, str(patient_id))
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir, exist_ok=True)

                output_location = os.path.join(output_subdir, "notes.pkl")
                patient_notes.to_pickle(output_location)


    logging.info("{} entries processed".format(total))

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
def test_merge(notes_path = "data/extracted_notes", vitals_path = "data/in-hospital-mortality",
               hadm2index_path =""
               ):

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

if __name__ == "__main__":
    extract_notes()
    # test_merge()