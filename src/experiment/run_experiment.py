import logging
import click
import os
from ..models.data_selection import KMeans_filtering
from ..data.make_dataset import importdb
import pickle

def main(input_filepath):
    ## read data
    if input_filepath.split('.')[-1]=='pkl':
        data = pickle.load(open(input_filepath, 'rb'))
    elif input_filepath.split('.')[-1]=='db':
        data = importdb(input_filepath)
    elif os.path.isfile(input_filepath):
        raise IOError('File is neither a Pickle dictionary nor SQLite3 database.')
    else:
        raise IOError('input_filepath not recognized as file.')

    for name, data_dict in data.items():
        X = data_dict['X']
        y = data_dict['y']
        ## introduce noise
        pass
        ## run data selection
        pass
        ## train
        pass
        ## report
        pass

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
