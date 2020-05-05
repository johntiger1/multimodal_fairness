from mimic3benchmark.readers import DecompensationReader

reader = DecompensationReader(dataset_dir='data/decompensation/train',
                              listfile='data/decompensation/train/listfile.csv')

print(reader.read_example(10))
print("keys??")
print("{} KEY is {}".format(

    "ICUSTAY_ID", reader.read_example(10)["name"].split("_")[0]

)) # the name has what we want. now we can go and join stuff with text