# Ugly hack modifying the relevant components of main.py and utils.py for in_hospital_mortality

# COMMAND: python -um mimic3models.in_hospital_mortality.gen_responses --data data/in-hospital-mortality/ --timestep 1.0  --network mimic3models/keras_models/lstm.py  --batch_size 8 --load_state mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch28.test0.286221665488.state  --output_dir mimic3models/in_hospital_mortality --dim 16  --depth 2 --dropout 0.3
# Assumes that you have the weights in the folder described (will be commited)

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import common_utils




# UGLY HACK FLAG, TODO: REFACTOR
TEST_ON_TRAIN = True


#CODE MODIFIED FROM utils.py
def load_data(reader, discretizer, normalizer, return_names=False):
    N = reader.get_number_of_examples()
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("patient_id,episode,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            assert(name.endswith("_timeseries.csv"))
            name = name[:-15]
            vals = name.split("_episode")
            assert(len(vals) == 2 and vals[0].isdigit() and vals[1].isdigit())
            f.write("{},{},{:.6f},{}\n".format(vals[0], vals[1], x, y))

# CODE MODIFIED FROM main.py

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
#parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
print(args)


#target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
# Not sure what this setting is, but author's didn't use it for optimal solution so I'm hard-coding it off
args_dict['target_repl'] = False
args_dict['target_repl_coef'] = 0

# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   "")
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}


loss = 'binary_crossentropy'
loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))






# ensure that the code uses test_reader
del train_reader

if TEST_ON_TRAIN:
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)
else:
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
ret = load_data(test_reader, discretizer, normalizer, return_names=True)

data = ret["data"][0]
labels = ret["data"][1]
names = ret["names"]

predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
predictions = np.array(predictions)[:, 0]
metrics.print_metrics_binary(labels, predictions)

if TEST_ON_TRAIN:
    path = os.path.join(args.output_dir, "train_predictions", os.path.basename(args.load_state)) + "_id_ep_fmt.csv"
else:
    path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + "_id_ep_fmt.csv"

save_results(names, predictions, labels, path)




