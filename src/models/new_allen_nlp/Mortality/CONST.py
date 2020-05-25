LOGGER_NAME = "decompensation_logger"

# now, we will set the stuff on the args appropriately (a seamless merge)

phenotyping_config = {


}

mortality_config = {

"train_data": "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
"dev_data" : "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv",
"data_type": "MORTALITY"
}

decomp_config = {

    "train_data": "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/train/listfile.csv",
    "dev_data": "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/test/listfile.csv",
    "data_type": "DECOMPENSATION"

}
configs = {"phenotyping_config": phenotyping_config,
           "mortality_config": mortality_config,
           "decomp_config": decomp_config}

def set_config(name, obj_to_set):
    for key, val in (configs[name]).items():
        print(f"{key} {val}\n")

        obj_to_set.__setattr__(key, val)
    return obj_to_set


