import argparse
import os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--exp_dir", type=str,
                        default="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/src/models/new_allen_nlp/experiments")

    args = parser.parse_args()
    args.serialization_dir = os.path.join(args.exp_dir,args.run_name)

    if not os.path.exists(args.serialization_dir):
        os.makedirs(args.serialization_dir, exist_ok=True)
    return args