'''

From the predictions.csv which are generated, we will load everything in and check

'''
import pandas as pd
import os
import logging
from sklearn import metrics
logger = logging.getLogger(__name__)
import allennlp.training

from tqdm.auto import tqdm

import CONST

def compute_auc(predictions_path: str=None, name: str="train"):
    # pd readcsv each of those, concat, then we pass in the data to AUCROC
    #
    all_preds_df = pd.DataFrame()
    for root, dirs, files in os.walk(predictions_path):
        for file in tqdm(files, total=500000//256):
            if str(file).startswith("predictions_"):
                # logger.critical(f"{file}\n")
                preds = pd.read_csv(os.path.join(root,file))
                all_preds_df = pd.concat((all_preds_df, preds), axis=0)

    all_preds_df["predictions"] = all_preds_df.apply(lambda row:  1 if row["probs_0" ] > 0.5 else 0, axis=1)
    print(metrics.roc_auc_score(all_preds_df["label_0"], all_preds_df["probs_0"] ))
    all_preds_df.to_csv( os.path.join(predictions_path,f"{name}_final_preds.csv"))
    # computed a different way: first, find the p/r curves, then find the area under it

    fpr,tpr, _ =metrics.roc_curve(all_preds_df["label_0"], all_preds_df["probs_0"] , 1)
    print(metrics.auc(fpr, tpr))
    print(len(all_preds_df))
    print(all_preds_df)

def load_serialized_model():
    pass




'''
attempts to load in a model, and then generate the predictions for the model.
The simple fact of the matter, is that we do indeed need to get all the data into memory first 
'''
def generate_predictions():
    import torch
    import pickle
    checkpoint = allennlp.training.Checkpointer(serialization_dir="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments/61-preproc-mort-get_train_preds/")

    model = torch.load("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments/61-preproc-mort-get_train_preds/best.th")
    path = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments/64-mort-no-preproc-all-preds-save-dsreader/dataset_reader.pkl"
    with open(path, "rb") as file:
        dataset_reader = pickle.load(file)
    # the model should know the vocab, indexers, and other info.

    # ideally, we would actually deserialize the args. Actually, it is irrelevant, since we just need to get the data into memory, and then run
    # it through the model.

    # run the model on the datas
    logger.setLevel(logging.CRITICAL)
    args = get_args.get_args()
    assert getattr(args, "run_name", None) is not None
    # args.run_name = "54-ihp-fixed-val-met"

    args.batch_size = 400
    args.train_data = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/train/listfile.csv"
    args.dev_data = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/test/listfile.csv"
    args.test_data = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/decompensation/test/listfile.csv"
    args.use_gpu = True
    args.lazy = False  # should be hardcoded to True, unless you have a good reason otherwise
    args.use_preprocessing = True
    args.device = torch.device("cuda:0" if args.use_gpu else "cpu")
    args.use_subsampling = True  # this argument doesn't really control anything. It is all in the limit_examples param
    args.limit_examples = 50000
    args.sampler_type = "balanced"
    # args.data_type = "MORTALITY"
    args.use_reg = False
    args.data_type = "MORTALITY"
    args.max_tokens = 1600
    args.get_train_predictions = True

    # 5 to 8 iterations per second for decomp pred
    #

    file_logger_handler = logging.FileHandler(filename=os.path.join(args.serialization_dir, "log.log"))
    file_logger_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(file_logger_handler)

    CONST.set_config(args.data_type, args)
    '''
    napkin math: 8s/iteration and then 500 000 / 256 => roughly 4 hours to run
    '''
    import time

    start_time = time.time()
    # mr = MortalityReader()
    # instances = mr.read("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv")
    # for inst in instances[:10]:
    #     print(inst)
    print("we are running with the following info")
    print("Torch version {} Cuda version {} cuda available? {}".format(torch.__version__, torch.version.cuda,
                                                                       torch.cuda.is_available()))
    # We've copied the training loop from an earlier example, with updated model
    # code, above in the Setup section. We run the training loop to get a trained
    # model.

    dataset_reader = build_dataset_reader(
        train_listfile=args.train_data,
        test_listfile=args.test_data,

        limit_examples=args.limit_examples, lazy=args.lazy, max_tokens=args.max_tokens,
        use_preprocessing=args.use_preprocessing,
        mode="train", data_type=args.data_type, args=args)

    train_data, dev_data = read_data(dataset_reader, args.train_data, args.dev_data)

    vocab = build_vocab(train_data + dev_data)

    # make sure to index the vocab before adding it
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    train_dataloader, dev_dataloader = build_data_loaders(dataset_reader, train_data, dev_data, args)

    pass


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger("aa")
    # my_out_handle = logging.StreamHandler()
    # logger.addHandler(my_out_handle)

    # don't use raw logging
    # logger.debug("this will cause a handler to be created")
    #
    # logger.warning("???")

    # make some

    # logger by default seems to use a stream handler which routes to std err

    # generate_predictions()
    experiments_dir = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments/"
    split = "train"
    predictions_path = os.path.join(experiments_dir, f"80-mort-pretrained/{split}")
    compute_auc(predictions_path=predictions_path, name=split)

