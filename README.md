
# Multimodal Fairness

Code for the paper "Exploring Text Specific and Blackbox Fairness Algorithms in Multimodal Clinical NLP"

[Best Paper Award at Clinical NLP 2020](https://clinical-nlp.github.io/2020/program.html)

# Environment setup
This is a complex repo with 3 main portions:
- AllenNLP: https://github.com/johntiger1/multimodal_fairness/tree/master/src/models/new_allen_nlp
- Fairness processing: 
	- https://github.com/johntiger1/multimodal_fairness/tree/master/IanFairnessHackery
	- https://github.com/johntiger1/multimodal_fairness/tree/master/mimic3models
- Ensemble and Physiological signals Modality classifier: https://github.com/johntiger1/multimodal_fairness/tree/master/ensemblemodels

Each of these could be run independently, and so we have 3 separate environments (although there is work to consolidate these dependencies)

## Installation
```
# create the conda environments
conda env create -f requirements.yml
conda create -n fairness_env python=3.8 
conda create -n timeseries_env python=3.8

# install the fairness depedencies in its env
conda activate fairness_env
pip install -r requirements.txt

# install the timeseries dependencies in its env
conda activate timeseries_env
-- fetch requirements from here: https://github.com/YerevaNN/mimic3-benchmarks/blob/master/requirements.txt -- 
pip install -r requirements.txt

```

## Fairness pipeline
Fairness is computed via the following steps:
1. Generate fairness_dict, mapping patients to their sensitive attributes `generate_sensitive.py`
2. Get results from text (unstructured) model, and pass them through: `merge_and_convert.py`

See: https://drive.google.com/drive/u/1/folders/16dZI4lfA9ORp-_z5CJOiFzThj39iIQbJ

This will bring everything to a episode/id/pred/label format. Then, you can evaluate the metrics using `evaluate_phenotype_preds.py`

## Data location
Data is available locally on: `./data`. It will also be available more widely, at e.g. `/scratch/gobi2/johnchen/...`

## Text 
Text model is run inside the `new_allen_nlp` folder: 
`https://github.com/johntiger1/multimodal_fairness/tree/master/src/models/new_allen_nlp`

At a high level this is a CNNEncoder, which is trained from scratch (i.e. does not use any pretrained LM). However, one emphasis of our paper is on training word embeddings from scratch, vs utilizing domain-relevant word embeddings vs domain relevant debiased word embeddings. 

## Tabular
Tabular model is the underlying 2019 *Nature* paper. This is a channel-wise LSTM classifier. 

## Ensemble
Ensemble is a sklearn (logistic regression) on the outputs of the text and tabular model. We train the logistic regression model on the outputs of the model, along with the final label. Note that the individual performances of the ensemble are fairly close; hence why we can perform a discrete ensembling step on just the outputs. 

# Fairness
Bulk of fairness code is: `https://github.com/johntiger1/multimodal_fairness/tree/master/mimic3models/fair_postprocess.py`

## Evaluation and Plotting

1. First, ensure you have predictions saved to disk for both text and tabular models, and the ensemble models. 
2. Plotting code is here: https://github.com/johntiger1/multimodal_fairness/tree/master/mimic3models/fair_postprocess.py
3. You need to install a different (conflicting set of requirements). Make sure you switch to a new virtual environment.
4. Edit the sourcecode of the fairlearn library; commenting out line 112 ("check_is_fitted(self.estimator)") of fairlearn/postprocessing/\_threshold_optimizer.py
5. Generate the file of sensitive attributes. This is a file which maps sensitive attributes to the cohort. Note that the cohort is mainly fixed by the mimic-3 benchmark itself.  
6. Run fair\_postprocess.py . It consists of two steps. First, run the LOAD command, which will map sensitive attributes for the purpose of fairness. This will produce an appropriate JSON dict. And then run the PLOT_ALL command to ingest all CSV of structured, unstructured, and ensemble data. You may need to run a "merge-coerce" script which will merge the unstructured predictions, and coerce them into the right, common format. The script to convert a csv into the standard format is IanFairnessHackery/merge\_and\_convert.py which uses the python2 environment (mmvenv, i.e. the environment that the baseline models use). Check the README's in the corresponding subdirectories for further details

An example of the PLOT_ALL command:
`PLOT_ALL7 IanFairnessHackery/john_results/biow2vec_mortality/train_final_preds_id_ep_fmt.csv IanFairnessHackery/john_results/biow2vec_mortality/test_final_preds_id_ep_fmt.csv mimic3models/in_hospital_mortality/train_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt.csv mimic3models/in_hospital_mortality/test_predictions/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt.csv ensemblemodels/bio2vec_mortality/default/train/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt+train_final_preds_id_ep_fmt\{\}_id_ep_fmt.csv ensemblemodels/bio2vec_mortality/default/test/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt+test_final_preds_id_ep_fmt\{\}_id_ep_fmt.csv IanFairnessHackery/john_results/Debiased_WE_ETH_4_Mortality/train_final_preds_id_ep_fmt.csv IanFairnessHackery/john_results/Debiased_WE_ETH_4_Mortality/test_final_preds_id_ep_fmt.csv ensemblemodels/debiased_WE_ETH_4_mortality/default/train/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt+train_final_preds_id_ep_fmt\{\}_id_ep_fmt.csv ensemblemodels/debiased_WE_ETH_4_mortality/default/test/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state_id_ep_fmt+test_final_preds_id_ep_fmt\{\}_id_ep_fmt.csv ETHNICITY HARD`
## Additional slides and documentation

Decomp results:
https://drive.google.com/drive/u/0/folders/1r7NamihCr8axiFs2m3gqJTwfBqggUGy4

Clinical NLP paper:
https://clinical-nlp.github.io/2020/program.html


## Additional resources
Original time series LSTM classifier: https://github.com/YerevaNN/mimic3-benchmarks
