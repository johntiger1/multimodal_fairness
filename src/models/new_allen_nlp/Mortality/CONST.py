LOGGER_NAME = "decompensation_logger"

# now, we will set the stuff on the args appropriately (a seamless merge)
decomp_config = {

    "key1" : 23,
    "key2": 24 #reinventing the json and config

}
args = lambda x: None
for key,val in (decomp_config).items():
    print(f"{key} {val}\n")

    args.__setattr__(key, val)

print(args)