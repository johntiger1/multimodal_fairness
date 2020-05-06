# May 5 
Finished all the preprocessing commands. Going to do the following:
1. Re-import the original notebook for dataprocessing.
2. Recall that interestingly, there are several satisfying solutions to use the notebook.
  1. we can get it to hook up running REMOTELY on my vector computer (SSH tunnelling)
  2. we can get it to hook up running REMOTELY on the cluster computers (Jupyter notebook; PHRI)
  3. We can simply use it locally on the vector machines; this is via VNC. (best approach/simplest)

Therefore, we will set up the notebook, in order to make our SQL queries to examine the right events.

Essentially, we simply perform the processing of the benchmark, in order to get preprocessed (better?: dubious) data. 

Q: doesn't it make more sense to just use ICUSTAY to directly link with the patients?
A: possibly. Instead, we have the X. The X is important since we have PATIENTS that are tracked longitudinally ; it doesn't make that much sense to track everything as a sequence of independent events!

Indeed, I think that explains everything. For mortality prediction: it is 48 hours. Note that everyone dies in the end. 

However, we *should* still be able to join things back and forth! In particular: we have specific ICUSTAYS and the notes correspond directly to those times! 

Indeed, because we want to do first-48 hour ICU mortality prediction, then that is reason enough for a) the time based approach (instead of joining directly on the ICUSTAY. Note that we can do various SQL optimization, like filter before join etc.). and b) the reason for why we don't just look at icustay, but instead at the patient. 


Timeseries, is kind of like the PURE data. We can also measure the trajectories then. 
Staged: classification + regression => First a network predicts whether or not the time will happen, and then next, another network predicts time; loss is only computed if the first network gets it wrong, essentially. 

The key takeaways:
- we understand how the data is preprocessed
- AND how we can train a model to use it
- that is for mortality, then we can do it for phenotyping, and decompensation prediction 

This github seems like exactly what I want: just merging the TF-IDF stuff on top of it. 
https://github.com/kaggarwal/ClinicalNotesICU

(defending work: fairness novelty. as well as more technical analysis of the word embeddings. Their work was not initially novel, because of work by Horng et al. and so we can argue we use better and more word embeddings (deep contextualized ones, for instance) )

But more word embeddings means we should stick to pytorch, OK.
And it would be good learning, anyways. (in particular if we want to extend the work later)

Also:
The readers pretty directly provide the data we want to work with. Features, and the label. 
But what about the timesteps? They are also all there! 

Therefore: 
N * T * K (where K is simply the number of variables we have, T is the number of timesteps, N is the number of rows/patients we have)

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
