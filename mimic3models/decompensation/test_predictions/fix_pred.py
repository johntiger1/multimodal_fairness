# Quick ugly hack script to convert the baseline test output into the format we need

# Set to path to predictions
READ_PATH = "./k_lstm.n128.dep1.bs8.ts1.0.chunk25.test0.0779094241263.state.csv"


assert(READ_PATH.endswith(".csv"))
WRITE_PATH = READ_PATH[:-4] + "_id_ep_fmt.csv"


with open(READ_PATH, 'r') as fr:
    with open(WRITE_PATH, 'w') as fw:
        fr.readline() #read header
        fw.write("id,episode,time,prediction,label\n")

        for line in fr:
            values = line.split(",")

            name = values[0]
            assert(name.endswith("_timeseries.csv"))
            name = name[:-15]
            id_and_ep = name.split("_episode")
            assert(len(id_and_ep) == 2 and id_and_ep[0].isdigit() and id_and_ep[1].isdigit())

            time = values[1]
            times = time.split(".")
            assert(len(times) == 2)
            assert(times[1] == "000000")
            time = times[0]

            fw.write("{},{},{},{},{}".format(id_and_ep[0], id_and_ep[1], time, values[2], values[3]))

