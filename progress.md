# May 5 
Finished all the preprocessing commands. Going to do the following:
1. Re-import the original notebook for dataprocessing.
2. Recall that interestingly, there are several satisfying solutions to use the notebook.
  1. we can get it to hook up running REMOTELY on my vector computer (SSH tunnelling)
  2. we can get it to hook up running REMOTELY on the cluster computers (Jupyter notebook; PHRI)
  3. We can simply use it locally on the vector machines; this is via VNC. (best approach/simplest)

Therefore, we will set up the notebook, in order to make our SQL queries to examine the right events.

Essentially, we simply perform the processing of the benchmark, in order to get preprocessed (better?: dubious) data. 

# May 4

Currently running:

python -m mimic3benchmark.scripts.validate_events data/root/ 

Started May 4, 4pm
Finished? Yes

Started may 4: 520pm
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
Finished? Yes

Started may 4: 940pm
python -m mimic3benchmark.scripts.split_train_and_test data/root/
Finished? (seems no progress bar)
