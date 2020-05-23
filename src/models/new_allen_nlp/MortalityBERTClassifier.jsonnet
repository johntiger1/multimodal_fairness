local bert_model = "albert-base-v1";

{
    "dataset_reader" : {
        "type": "MortalityReader",
        "max_tokens": 256,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": 256
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
                "max_length": 128
                # actually they do!
            }
        },
        "lazy": false
    },
    "train_data_path": "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
    "validation_data_path": "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv",
    "model": {
        "type": "MortalityClassifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model,
                    "max_length": 128
                }
            }
        },
            "encoder": {
            "type": "bert_pooler",
            "pretrained_model": bert_model,
            "requires_grad": false
        }
    },
    "data_loader": {
        "batch_size": 32
        #found this has an effect
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 50,
        "cuda_device": 0,
        "patience": 5,
        "validation_metric": "+auc"
    }
}