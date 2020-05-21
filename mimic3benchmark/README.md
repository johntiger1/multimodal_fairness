Ian's quick and dirty notes:

New design, just run the authors's test code with minimal modifications so that we can also get the predictions on the training data.
Once their code is finished running, run a postprocessing script to clean their .csv into the format we're using

Command to run in-hospital mortality; To run on training data add the flag --test_on_train

python -um mimic3models.in_hospital_mortality.gen_responses --mode test --data data/in-hospital-mortality/ --timestep 1.0  --network mimic3models/keras_models/channel_wise_lstms.py  --batch_size 8 --load_state mimic3models/in_hospital_mortality/keras_states/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state  --output_dir mimic3models/in_hospital_mortality --dim 8  --depth 1 --dropout 0.3 --size_coef 4



Command to run phenotyping; To run on training data add the flag --test_on_train

python -um mimic3models.phenotyping.gen_responses --mode test --network mimic3models/keras_models/channel_wise_lstms.py --load_state mimic3models/phenotyping/keras_states/nr6k_channel_wise_lstms.n16.szc8.0.d0.3.dep1.bs64.ts1.0.epoch49.test0.348234337795.state --dim 16 --timestep 1.0 --depth 1 --dropout 0.3 --batch_size 64 --output_dir mimic3models/phenotyping --data data/phenotyping/ --size_coef 8



Command to run decompensation; again use flag --test_on_train to get predictions on train data

python -um mimic3models.decompensation.gen_responses --mode test --network mimic3models/keras_models/channel_wise_lstms.py --load_state mimic3models/decompensation/keras_states/nrk_channel_wise_lstms.n16.szc8.0.dep1.dsup.bs32.ts1.0.chunk6.test0.0810981076094.state --dim 16 --timestep 1.0 --depth 1 --batch_size 32 --output_dir mimic3models/decompensation --data data/decompensation/ --size_coef 8 --deep_supervision



