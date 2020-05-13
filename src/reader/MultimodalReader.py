'''
This is a multimodal reader, it also reads in the text modality.
For now, we wil assume the text data is already broken up appropriately

Otherwise, we could simply reindex the notes_table each time, but then that requires 
a) keeping a copy of the notes_table around always (centralization)
b) repeated querying

For POC right now, we will still have a PER_PATIENT sql lookup

In the future, we will most likely support this API:
patient: 
- vital data => in .csv
- text data => also in .csv
'''
from mimic3benchmark.readers import Reader
import os 
import pandas as pd
import numpy as np
import logging


class MultiInHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, text_data_dir="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes", listfile=None, period_length=48.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self.patient2idx = {}
        for idx, elt in enumerate (self._data):
            pat_id = elt.split("_")[0]
            self.patient2idx[int(pat_id)]  = idx
        self._data = [line.split(',') for line in self._data]

        # self.patient2idx = {pat_id:idx for a.split("_") for entry in self._data }
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length
        self.textdata_dir = text_data_dir

    '''finds the idx for the given patient'''
    def patient2idx(self, pat_id):

        # for i,
        # pass
        pass

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)
        
        patient_id = (name.split("_")[0])
        hadm_id = None #TODO, fill this in with the mapping
        episode = None # episode1 => need a re; int(name.split("_")[1])
        relevant_patient_notes = pd.read_pickle(os.path.join(self.textdata_dir, patient_id, "notes.pkl"))
        text = relevant_patient_notes.iloc[0] #TODO: fill this in with the mapping
        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name, 
                "text": text}



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)


    dual_reader  = MultiInHospitalMortalityReader (dataset_dir='data/in-hospital-mortality/test',
                                                   text_data_dir="/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes",
                              listfile='data/in-hospital-mortality/test/listfile.csv')
    idx = dual_reader.patient2idx[1819]
    datum = dual_reader.read_example(idx )
    logger.info(datum)
    pass
