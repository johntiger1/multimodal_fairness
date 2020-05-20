# More ugly hackery, now even lazier!
# python -um mimic3models.phenotyping.gen_responses --network mimic3models/keras_models/lstm.py --load_state mimic3models/phenotyping/keras_states/k_lstm.n256.d0.3.dep1.bs8.ts1.0.epoch20.test0.348495262943.state --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/phenotyping --data data/phenotyping/
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import imp
import re
import os

from mimic3benchmark.readers import PhenotypingReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils

from keras.callbacks import ModelCheckpoint, CSVLogger



from mimic3models import common_utils
import threading
import random
import os

# UGLY HACK TODO: Refactor
TEST_ON_TRAIN = True

PRED_TASKS = {
    1: "Acute and unspecified renal failure",
    2: "Acute cerebrovascular disease",
    3: "Acute myocardial infarction",
    4: "Cardiac dysrhythmias",
    5: "Chronic kidney disease",
    6: "Chronic obstructive pulmonary disease and bronchiectasis",
    7: "Complications of surgical procedures or medical care",
    8: "Conduction disorders",
    9: "Congestive heart failure",
    10: "nonhypertensive",
    11: "Coronary atherosclerosis and other heart disease",
    12: "Diabetes mellitus with complications",
    13: "Diabetes mellitus without complication",
    14: "Disorders of lipid metabolism",
    15: "Essential hypertension",
    16: "Fluid and electrolyte disorders",
    17: "Gastrointestinal hemorrhage",
    18: "Hypertension with complications and secondary hypertension",
    19: "Other liver diseases",
    20: "Other lower respiratory disease",
    21: "Other upper respiratory disease",
    22: "Pleurisy",
    23: "pneumothorax",
    24: "pulmonary collapse",
    25: "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)"
}


class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer, batch_size,
                 small_part, target_repl, shuffle, return_names=False):
        self.batch_size = batch_size
        self.target_repl = target_repl
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_data(reader, discretizer, normalizer, small_part)

        self.steps = (len(self.data[0]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_data(self, reader, discretizer, normalizer, small_part=False):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000
        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if (normalizer is not None):
            data = [normalizer.transform(X) for X in data]
        ys = np.array(ys, dtype=np.int32)
        self.data = (data, ys)
        self.ts = ts
        self.names = names

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[None] * N, [None] * N]
                tmp_names = [None] * N
                tmp_ts = [None] * N
                for i in range(N):
                    tmp_data[0][i] = self.data[0][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_names[i] = self.names[order[i]]
                    tmp_ts[i] = self.ts[order[i]]
                self.data = tmp_data
                self.names = tmp_names
                self.ts = tmp_ts
            else:
                # sort entirely
                X = self.data[0]
                y = self.data[1]
                (X, y, self.names, self.ts) = common_utils.sort_and_shuffle([X, y, self.names, self.ts], B)
                self.data = [X, y]

            self.data[1] = np.array(self.data[1])  # this is important for Keras
            for i in range(0, len(self.data[0]), B):
                x = self.data[0][i:i+B]
                y = self.data[1][i:i+B]
                names = self.names[i:i + B]
                ts = self.ts[i:i + B]

                x = common_utils.pad_zeros(x)
                y = np.array(y)  # (B, 25)

                if self.target_repl:
                    y_rep = np.expand_dims(y, axis=1).repeat(x.shape[1], axis=1)  # (B, T, 25)
                    batch_data = (x, [y, y_rep])
                else:
                    batch_data = (x, y)

                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names, "ts": ts}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()


def save_results(names, ts, predictions, labels, dir, filename):
    n_tasks = 25
    common_utils.create_directory(os.path.dirname(dir))

    for i in range(1,n_tasks + 1):
        with open(os.path.join(dir, PRED_TASKS[i]+filename), 'w') as f:
            header = ["id", "episode", "prediction", "label"]
            header = ",".join(header)
            f.write(header + '\n')
            for name, t, pred, y in zip(names, ts, predictions, labels):
                assert (name.endswith("_timeseries.csv"))
                name = name[:-15]
                vals = name.split("_episode")
                assert (len(vals) == 2 and vals[0].isdigit() and vals[1].isdigit())
                line = vals
                #line += ["{:.6f}".format(t)] # We don't need time in the output
                line += ["{:.6f}".format(pred[i-1])]
                line += [str(y[i-1])]
                line = ",".join(line)
                f.write(line + '\n')


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/phenotyping/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
print(args)

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_listfile.csv'))

val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                               listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ph'
args_dict['num_classes'] = 25
args_dict['target_repl'] = target_repl

# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
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


# Build data generators
train_data_gen = BatchGen(train_reader, discretizer,
                                normalizer, args.batch_size,
                                args.small_part, target_repl, shuffle=True)
val_data_gen = BatchGen(val_reader, discretizer,
                              normalizer, args.batch_size,
                              args.small_part, target_repl, shuffle=False)

if args.mode == 'train':
    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.PhenotypingMetrics(train_data_gen=train_data_gen,
                                                      val_data_gen=val_data_gen,
                                                      batch_size=args.batch_size,
                                                      verbose=args.verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=train_data_gen.steps,
                        validation_data=val_data_gen,
                        validation_steps=val_data_gen.steps,
                        epochs=n_trained_chunks + args.epochs,
                        initial_epoch=n_trained_chunks,
                        callbacks=[metrics_callback, saver, csv_logger],
                        verbose=args.verbose)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_data_gen
    del val_data_gen

    if TEST_ON_TRAIN:
        test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_listfile.csv'))
    else:
        test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'))

    test_data_gen = BatchGen(test_reader, discretizer,
                                   normalizer, args.batch_size,
                                   args.small_part, target_repl,
                                   shuffle=False, return_names=True)

    names = []
    ts = []
    labels = []
    predictions = []
    for i in range(test_data_gen.steps):
        print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
        ret = next(test_data_gen)
        x = ret["data"][0]
        y = ret["data"][1]
        cur_names = ret["names"]
        cur_ts = ret["ts"]
        x = np.array(x)
        pred = model.predict_on_batch(x)
        predictions += list(pred)
        labels += list(y)
        names += list(cur_names)
        ts += list(cur_ts)

    metrics.print_metrics_multilabel(labels, predictions)
    dirname = os.path.join(args.output_dir, "test_predictions")
    filename = os.path.basename(args.load_state) + "_id_ep_fmt.csv"
    if TEST_ON_TRAIN:
        filename = "train_" + filename
    save_results(names, ts, predictions, labels, dirname, filename)

else:
    raise ValueError("Wrong value for args.mode")
