Ian's quick and dirty notes:

Benchmark model predictions have been commited as a .csv under mimic3models/TASK_NAME_HERE/test_predictions/

I will commit the weights I used/trained

You shouldn't need to generate the predictions again, but just in case:

NOTE: I've added a hack hard-coded flag to run test on the train data (go to code and set TEST_ON_TRAIN to false). Will refactor when there is time

To generate the in-hospital mortality test results run (update --load_state if you want to use a different weights checkpoint):

python -um mimic3models.in_hospital_mortality.gen_responses --data data/in-hospital-mortality/ --timestep 1.0  --network mimic3models/keras_models/lstm.py  --batch_size 8 --load_state mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch28.test0.286221665488.state  --output_dir mimic3models/in_hospital_mortality --dim 16  --depth 2 --dropout 0.3

To generate the phenotyping test results run:

python -um mimic3models.phenotyping.gen_responses --network mimic3models/keras_models/lstm.py --load_state mimic3models/phenotyping/keras_states/k_lstm.n256.d0.3.dep1.bs8.ts1.0.epoch20.test0.348495262943.state --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/phenotyping --data data/phenotyping/

The decompensation is a bit of an uglier hack since I wanted to do it quickly, and it takes a while for the tests to run. First execute:

   python -um mimic3models.decompensation.main --network mimic3models/keras_models/lstm.py --dim 128 --timestep 1.0 --depth 1 --mode test --batch_size 8 --output_dir mimic3models/decompensation --load_state mimic3models/decompensation/keras_states/k_lstm.n128.dep1.bs8.ts1.0.chunk25.test0.0779094241263.state
   
This will generate a .csv file under the test_results folder. There there's a python script that can read the .csv and split the first column as needed. Just open mimic3models/decompensation/test_predictions/fix_pred.py, update the hardcoded path to the file, and run it. It will generate the required.csv (filename will be suffixed with _id_ep_fmt)






