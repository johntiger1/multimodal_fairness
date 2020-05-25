import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    return args