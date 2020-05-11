from mimic3benchmark.readers import DecompensationReader, InHospitalMortalityReader
import pandas as pd
import logging
import glob
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
def extract_notes(data_path="data/root"):

    for filepath in glob.iglob(data_path):
        print(filepath)



if __name__ == "__main__":
    extract_notes()
