# May 14
Doubling down:

AllenNLP: datasetReader and Model
1. DatasetReader: defines a read method
  read: yields Instances, which are composed of named fields.
2. Model: defines a forward; just like a regular PyTorch module

(as expected)

A little more depth:
 1. When we have Instances, they are composed of Fields. These Fields are internally used by the model, which does the standard processing (text => tokenization => embedding look-up). Note that text => tokenization is not directly handled by either the model OR the reader; rather, this is served by a config. 
 2. 
 
Strong choices: loss is computed right in the *forward* pass of the model! More on that later...
( directly goes against the 4-part PyTorch coding approach )



# May 13
OK, switching to AllenNLP, since I want to try a new library.

Fairseq Prog:
The goal is limited, and achievable. Essentially, we want to do: make a fairseq ConvEncoder, that will simply predict the target sequence (0/1), from the given vocab. of texts. Therefore, all we need to do is build the dictionary, and then leverage word embeddings etc.

We will need to either build the dict ourselves, or use the built-in preproc tool. This preproc tool: will auto build it from corpora. But we need the corpora in the right format. 

FairSeq is NOT plug and play, we will need to implement and satisfy an implementation for an interface, defining a task, architecture, and so forth. 
___

Essentially, we should have a data reader, which iterates over both streams of data. And then, provide this data to a model, which will then train.

And in general, just need to build a simple LSTM.

Relevant libraries and packages:
- torchtext
- pytorch-nlp
- huggingface

For these last two: changes to their built-in models (on a local version) *should* also be reflected in the local version as well! 
https://github.com/allenai/allennlp-models
- allennlp
- fairseq 
- pytorch-nlp (*just* the CNN encoder)

Therefore, we will spend two hours investigating fairseq. One thing is that it is obtuse to begin. But this time may be different...

Instead, let us try and use allennlp, which has an attractive and useful tutorial. Anything that has a pre-trained model, must also provide us the functionality to dynamically edit and load it from scratch locally

# May 10

# may 10 Appendix 
- Gensim is awesome, really reduces barriers to leveraging things. 
- And just has so much functionality and use cases already in the box
- we can use a full model, or simply a keyed-vectors. The Keyed-vectors allows us to simply look up stuff, treating it as fixed features for lookup!
- Therefore, we will use gensim FASTTEXT and then not have to worry about OOV words
- Tempered Gensim: it is good, and active, but not production/industrial quality. Lots of issues with integrations with FastText
However, we DO have a nice, most simple basic model. 

OK, so the paths forward. 
1. Use the newer version of gensim, curl/wget the zip file, unzip, and then load the bin. OK
2. Keep trying with the OG version, and trying to co-erce the loading. We can even try going to FastText *github* and then working with those

# May 9
Two possible pathways:
1. Text focus: how do the word embeddings play with the performance. In particular, if we use w2vec embeddings, we *must* switch to a neural network approach. This risks broadening the scope to an empirical analysis of word embeddings (no it doesn't. But `Research question: a general framework for employing word embeddings with traditional neural network architectures. ` IS a possibility (_option_) , but we don't need to exercise it. In particular, we can either drop the BOW model completely, or simply include them as additional results, without focusing in-depth on the performance differences. Additionally/finally, by switching to word embeddings, then we also have all NN-based fairness approaches, which are transferrable to Transformer stuff. 

Another possibility could be empirical fairness eval when switching from ML models to neural models. How does the language modelling process produce useful, zero-tuning embeddings for a variety of classification tasks. 

2. Principled focus: just trying to set up the data pipeline. But 1 will eventually need to be addressed regardless. 

Important:
1. debiased word embeddings. 
and think about principled approaches to leveraging word embeddings. If out of vocab, then what happens? We _could_ use the scikit learn Log reg to transform it, and then it would just give us OOV, which is fine. And we can see the performance. yes, this is a good first approach. 

Listfile.csv simply is a pointer to the actual file with the data. But otherwise it is meaningless, we will need to go and open the OG data to get HADM info and so forth
It seems they are partitioned so that the patients don't appear in both the train and test split. And also, one visit may be discounted/discarded *for one task*

Mostly figured out, but can we have multiple icustay for a single hosp adm? Possibly

For a single hospital stay, seems like there can be multiple ICU stays. -- try running a group by query!
https://mimic.physionet.org/mimictables/transfers/

The subject IDs are LITERALLY the subject ids in MIMIC too!

# May 7
OK, so now let's just work on it easily. Essentially, we can run the baseline, and see what result we get. That is quite simple! 

Why this is simple:
- we are doing time series prediction. we just read in the csv files, as readers, to get the features, and then we need to do the joins to get all the notes. And we can leverage Haoran's code to do it.

Note that https://github.com/kaggarwal/ClinicalNotesICU only does the 3 tasks (and NOT phenotyping). Therefore, just doing phenotyping is by itself novel, plus the fairness.  We also provide open-source code which allows for flexible generation and combination of text with the mimic data, allowing for reproducible builds, and further opening the door for future joint text-numeric analysis. 

Potential blockers:
1. want to get the joins in one shot: i.e. one framework for joining the notes and the outcome, for all things, as per Haoran's code. One thing is that a time series immediately destroys the nice matching, since we can have multiple correspondences to a single event. 

yet here is another perspective. We can ensure we do not have any note from AFTER the actual event.

Potential issues: in the note, it says Patient died. 
But the actual mortality event is not recorded for several hours later. 
Is that an issue in practice? Is there causal leakage? If so, the check if KAgg also addresses it.

2. ensuring no causal leakage, as per KAgg and Haoran 

One other area, is that we could try leveraging the clinical word embeddings, to see what happens. 

# May 6 
Mainly was working on the seq2seq and other fairness stuff. 

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
f

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
