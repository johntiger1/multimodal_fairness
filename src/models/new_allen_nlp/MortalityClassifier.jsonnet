{
    "dataset_reader" : {
        "type": "MortalityReader",
        "max_tokens": 999,
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        },
    },
    "train_data_path": "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
    "validation_data_path": "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/test/listfile.csv",
    "model": {
        "type": "MortalityClassifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300
                }
            }
        },
        "encoder": {
            "type": "cnn",
            "embedding_dim": 300
        }
    },
    "data_loader": {
        "batch_size": 1024,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 50,
        "cuda_device": 0,
        "patience": 5,
        "validation_metric": "+auc"
    }
}
