from mimic3benchmark.readers import DecompensationReader, InHospitalMortalityReader
import pandas as pd
import logging

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

'''
def extract_notes():

    reader = InHospitalMortalityReader(dataset_dir='data/in-hospital-mortality/train',
                                       listfile='data/in-hospital-mortality/train/listfile.csv')
    total_examples = reader.get_number_of_examples()
    for i in range(total_examples):
        reader.read_example(i)
    # print("we have 100k indices, and they get split between train and test. ")
    # print("we also have different episodes split as well")
    # # print("Contains all the pertinent info for rejoining everything")
    # print(reader.read_example(10))
    #
    # print("so we have this 10th example. Now, what do we do to it?")
    # print(reader.read_example(10)["name"])
    # patient_id = reader.read_example(10)["name"].split("_")[0]
    # MIMIC_ROOT = "data/root/train/"
    # MIMIC_og_data_ROOT = "data/physionet.org/files/mimiciii/1.4/"
    # notes_table = "NOTEEVENTS.csv"
    # import os
    #
    # with open(os.path.join(MIMIC_ROOT, patient_id, "stays.csv"), "r") as file:
    #     print("finding relevant info for {}".format(patient_id))
    #     entries = []
    #     for line in file:
    #         stuff = line.split(",")
    #         print(stuff)
    #         entries.append(stuff[0:3])
    #     entries = entries[1:]
    # pass
    #



if __name__ == "__main__":
    extract_notes()
