Logistic regression ensemble is logistic_regression.py; in order to run must first have the results of the structured 
and unstructured data model predictions in the _id_ep_fmt.csv

To get the structured data predictions see the README in mimic3models

To get the unstructured data predictions, first TODO (get John's unstructured results from the Google Drive), then get 
the .csv's into the correct format by using the IanFairnessHackery/merge_and_convert.py script (give it the path to
John's results and it'll create a new file in the same directory in the new format)

Once you have all of these files in the correct format for your train data and test data, then you can go the root
directory and run

python2 -um ensemblemodels.logistic_regression 
--mode \<M or D or P depending on Mortality, Decomp. or Phen.\> 
--str_path \<PATH TO STRUCTURED TRAIN DATA\> 
--ustr_path \<PATH TO UNSTRUCTURED TRAIN DATA\> 
--test_str_path \<PATH TO STRUCTURED TEST DATA\>  
--test_ustr_path \<PATH TO UNSTRUCTURED TEST DATA\>
 --outdir \<PATH TO DIRECTORY IN WHICH TO SAVE PREDICTIONS\>