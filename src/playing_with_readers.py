from mimic3benchmark.readers import DecompensationReader, InHospitalMortalityReader

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
import os

with open(os.path.join(MIMIC_ROOT, patient_id, "stays.csv"), "r") as file:
    print("finding relevant info for {}".format(patient_id))
    for line in file:
        print(line)
    print("why are there two things in here? Because there are two episodes")
    print("the reason is that. the mortality prediction only uses 1 episode. The reason is that they may remove or invalidate parts of it for the mortality prediction"
          "For instance! the stay may have been < 48 hours"
          "but the decompensation prediction episode for examples uses 2 episodes")


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