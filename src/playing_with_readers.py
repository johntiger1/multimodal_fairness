from mimic3benchmark.readers import DecompensationReader, InHospitalMortalityReader
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("hello")
# reader = DecompensationReader(dataset_dir='data/decompensation/train',
#                               listfile='data/decompensation/train/listfile.csv')

reader =InHospitalMortalityReader(dataset_dir='data/in-hospital-mortality/train',
                              listfile='data/in-hospital-mortality/train/listfile.csv')

print("we have 100k indices, and they get split between train and test. ")
print("we also have different episodes split as well")
# print("Contains all the pertinent info for rejoining everything")
print(reader.read_example(10))

print("so we have this 10th example. Now, what do we do to it?")
print(reader.read_example(10)["name"])
patient_id = reader.read_example(10)["name"].split("_")[0]
MIMIC_ROOT = "data/root/train/"
MIMIC_og_data_ROOT = "data/physionet.org/files/mimiciii/1.4/"
notes_table = "NOTEEVENTS.csv"
import os

with open(os.path.join(MIMIC_ROOT, patient_id, "stays.csv"), "r") as file:
    print("finding relevant info for {}".format(patient_id))
    entries = []
    for line in file:
        stuff = line.split(",")
        print(stuff)
        entries.append(stuff[0:3])
    entries = entries[1:]
    print("why are there two things in here? Because there are two episodes")
    print("the reason is that. the mortality prediction only uses 1 episode. The reason is that they may remove or invalidate parts of it for the mortality prediction"
          "For instance! the stay may have been < 48 hours"
          "but the decompensation prediction episode for examples uses 2 episodes")
    # subj_id =
    # hadm_id = bb
    # icustay_id = cc
    print("now, we will do some interesting joining!")
    df = pd.read_csv(os.path.join(MIMIC_og_data_ROOT, notes_table))
    df.CHARTDATE = pd.to_datetime(df.CHARTDATE)
    df.CHARTTIME = pd.to_datetime(df.CHARTTIME)
    df.STORETIME = pd.to_datetime(df.STORETIME)

    df2 = df[df.SUBJECT_ID.notnull()]
    df2 = df2[df2.HADM_ID.notnull()]
    df2 = df2[df2.CHARTTIME.notnull()]
    df2 = df2[df2.TEXT.notnull()]

    df2 = df2[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]

    del df
    notes_df = df2
    specific_visit = list(map(lambda x: int (x), entries[0]))
    print(notes_df[(notes_df['SUBJECT_ID'] == specific_visit[0]) & (notes_df['HADM_ID'] == specific_visit[1])])
    logging.debug("Moving to jupyter notebook for max understanding")
# with open(os.path.join(MIMIC_ROOT, str(99999), "stays.csv"), "r") as file:
#     print("finding relevant info for {}".format(99999))
#     for line in file:
#         print(line)

# print("The")
# print("{} KEY is {}".format(
#
#     "ICUSTAY_ID", reader.read_example(10)["name"].split("_")[0]
#
# )) # the name has what we want. now we can go and join stuff with text

# print("one should note that the table is formed in the following manner: (it is time series)")
# print(reader.read_example(10)["X"].shape)
# print(len(reader.read_example(10)["header"]))

# now, let us try getting the actual stay information:


# now, we will try combining and reading everything