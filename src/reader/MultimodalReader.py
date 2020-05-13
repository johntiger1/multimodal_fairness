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


class MultiInHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, text_data_dir="", listfile=None, period_length=48.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length
        self.textdata_dir = text_data_dir

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
        
        patient_id = int(name.split("_")[0])
        hadm_id = None #TODO, fill this in with the mapping
        episode = int(name.split("_")[1])
        relevant_patient_notes = pd.read_pickle(os.path.join(self.textdata_dir, patient_id, "notes.pkl"))
        text = relevant_patient_notes.iloc[0] #TODO: fill this in with the mapping
        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name, 
                "text": text}