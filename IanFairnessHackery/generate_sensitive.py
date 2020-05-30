# Ugly hacky script to create .csv file to map from patient ID to all sensitive attributes

import pandas as pd

PATH_TO_BENCHMARK_STAYS = "../data/root/all_stays.csv"
PATH_TO_MIMIC_ADMISSIONS = "/home/administrator/00Projects/Fairness/MIMIC_III/MIMIC_III/ADMISSIONS.csv"
#PATH_TO_MIMIC_ADMISSIONS = "/h/shossain/multimodal_fairness/data/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv"

# Define data processing helpers:
# Aggregator such that if element of group is different, then UNKNOWN is returned, else value is returned
def unk_if_diff(x):
    default = None
    for i, val in enumerate(x):
        if i == 0:
            default = val
        else:
            if val != default:
                return "UNKNOWN"
    return default

# Shorten things to first - or / if applicable
def clean(x):
    if pd.isna(x):
        return "UNKNOWN"
    elif x in ["NOT SPECIFIED", "UNOBTAINABLE", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER", "UNKNOWN (DEFAULT)"]:
        return "UNKNOWN"
    elif x == "HISPANIC OR LATINO":
        return "HISPANIC"

    def truncate(x, pattern):
        ind = x.find(pattern)
        if ind != -1:
            return x[:ind]
        return x

    x = truncate(x,'-')
    x = truncate(x,'/')
    return x.strip()

#count = benchmark_df["SUBJECT_ID"].value_counts()
#print(count[count > 2])
#print(benchmark_df.loc[benchmark_df["SUBJECT_ID"] == 27374])
#benchmark_df.loc[benchmark_df["SUBJECT_ID"] == 27374,"ETHNICITY"] = "UNKNOWN/NOT SPECIFIED"
#benchmark_df.drop_duplicates(inplace=True)

#count = benchmark_df["SUBJECT_ID"].value_counts()
#print(count[count > 1].index)


def print_db(sensitive_dataframe):
    # My sanity checking
    print("Ethnicities in data:")
    print(sensitive_dataframe['ETHNICITY'].value_counts())

    print("Sex in data:")
    print(sensitive_dataframe['GENDER'].value_counts())

    print("Insurance in data:")
    print(sensitive_dataframe['INSURANCE'].value_counts())

    print("Religion in data:")
    print(sensitive_dataframe['RELIGION'].value_counts())

    print("Marital status in data:")
    print(sensitive_dataframe['MARITAL_STATUS'].value_counts())


# Load sensitive features from benchmark
benchmark_df = pd.read_csv(PATH_TO_BENCHMARK_STAYS)
benchmark_df = benchmark_df[["SUBJECT_ID","ETHNICITY","GENDER"]]

# Process to ensure SUBJECT_ID unique, and truncate descriptors
benchmark_df = benchmark_df.groupby("SUBJECT_ID").agg(unk_if_diff)
benchmark_df= benchmark_df.applymap(clean)
#print(benchmark_df)

# Load sensitive features from mimic, repeat processing
mimic_df = pd.read_csv(PATH_TO_MIMIC_ADMISSIONS, engine="python")
mimic_df = mimic_df[["SUBJECT_ID","INSURANCE","RELIGION", "MARITAL_STATUS"]]
mimic_df = mimic_df.groupby("SUBJECT_ID").agg(unk_if_diff)
mimic_df = mimic_df.applymap(clean)
#print(mimic_df)

# Do a join to get all of the sensitive attributes in a single dataframe
joined = benchmark_df.merge(mimic_df, on="SUBJECT_ID", how="inner", validate="one_to_one")
#print(joined)
joined.to_csv("full_detail_sensitive.csv")


print("ORIGINAL CLASSES")
print_db(joined)

original = joined.copy()

# Create first version of sensitive attributes - merge ethnicities into WHITE & NON_WHITE & save
joined["ETHNICITY"] = joined["ETHNICITY"].apply(lambda x: x if x == "WHITE" else "NON_WHITE")
joined.to_csv("sensitive_bin_eth.csv")


# Create first version of sensitive attributes - merge ethnicities into WHITE & NON_WHITE & save
joined = original.copy()
joined["INSURANCE"] = joined["INSURANCE"].apply(lambda x: x if x == "Medicare" else "Non-Medicare")
joined.to_csv("sensitive_bin_ins.csv")

# Create second version of sensitive attributes - bin ethnicities into 5 groups & save
joined = original.copy()
joined["ETHNICITY"] = joined["ETHNICITY"].apply(lambda x: x if x in ["WHITE", "BLACK","HISPANIC","ASIAN"] else "OTHER")
# Save results
joined.to_csv("sensitive_5_eth.csv")


# Create third version of sensitive attributes - LOSES DATA! Removes all with UNK ethnicity
joined = joined[joined.ETHNICITY != "OTHER"]
joined.to_csv("partial_sensitive_4_eth.csv")
print("4 ethnicity variant")
print_db(joined)

# Create fourth version of sensitive attributes - LOSES DATA! Removes all with UNK Insurance
# and merged medicaid with government
joined = joined[joined.INSURANCE != "UNKNOWN"]
joined["INSURANCE"] = joined["INSURANCE"].apply(lambda x: x if x in ["Private", "Medicare","Self Pay"] else "Gov/Medicaid")
joined.to_csv("partial_sensitive_all_4.csv")
print("4 ethnicity/insurance variant")
print_db(joined)

