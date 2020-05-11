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
    for dir, subdir, file in os.walk(data_path):
        total+=1
        for split in tqdm(subdir, total=100000):
            if split.isdigit():
                patient_id = int(split)

                with open(os.path.join(dir, split, "stays.csv")) as stays_file:
                    stays_file.readline() #consume the first line
                    hadm_id = int(stays_file.readline().split(",")[1])
                relevant_notes_idx = (notes_table["HADM_ID"] == hadm_id) &\
                                     (notes_table["SUBJECT_ID"] == patient_id)

                output_subdir = os.path.join(output_dir, str(patient_id))
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir, exist_ok=True)

                output_location = os.path.join(output_subdir, "notes.pkl")
                notes_table[ relevant_notes_idx].to_pickle(output_location)


        logging.info("{} entries processed".format(total))




if __name__ == "__main__":
    extract_notes()
