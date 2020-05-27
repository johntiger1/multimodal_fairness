generate_sensitive.py: 

Ugly hacky script to create .csv file to map from patient ID to all sensitive attributes
Just update the two hard-coded paths (if the data folder is already generated and in the default place, then you only 
have to update PATH_TO_MIMIC_ADMISSIONS)


To generate baseline model predictions, follow README in mimic3models, then run fix_pred.py


merge_and_convert.py

Ugly hacky script to convert (and potentially merge) John's unstructured data results into the episode/id/pred/label format.

For mortality set mode to M and give path to the .csv

For phenotyping set mode to P and give path to folder of John's results

python2 merge_and_convert.py --mode M --inpath ./john_results/final_preds_mort.csv

python2 merge_and_convert.py --mode M --inpath ./john_results/train_final_preds_mort.csv

python2 merge_and_convert.py --mode P --inpath ./john_results/Phenotyping/test/
python2 merge_and_convert.py --mode P --inpath ./john_results/Phenotyping/train/



evaluate_phenotype_preds.py

Script to read in folder of phenotyping predictions and calculate AUC for each, as well as macro/micro AUC over all:

python2 -um IanFairnessHackery.evaluate_phenotype_preds \< PATH TO DIR OF PHENOTYPE PREDICTIONS \>

Note that the script assumes that each file is prefixed with the name of it's task