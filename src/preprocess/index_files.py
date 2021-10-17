import os
import pandas as pd
import numpy as np
import dask.bytes
import argparse

from concurrent.futures import ProcessPoolExecutor, wait
import datetime
import sys

_PREPROCESS_DIR = os.path.abspath(__file__).split('/')[:-1]
_SRC_DIR = os.path.join('/',*_PREPROCESS_DIR[:-1])
sys.path.append(_SRC_DIR)

# # Native packages
import utils

utils.disable_warnings()
logger = utils.get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--datafolder',
    required=True,
    default=None,
    help='Folder containing a list of files to process'
)
parser.add_argument('--destination',
    default=None,
    help='Destination of index file'
)
parser.add_argument('--instrument',
    default='mdi',
    help='Instrument to process')
parser.add_argument('--nworkers',
    default=4,
    help='Number of parallel workers to use')
parser.add_argument('--indexfile',
    default='index.csv',
    help='Name of output index csv file')
parser.add_argument('--extension',
    default='.fits',
    help='File extension to use')
parser.add_argument('--prefix',
    default=None,
    help='Prefix that files need to contain to be considered')
parser.add_argument('--compression',
    default=None,
    help='File compression (i.e. gzip)')    



def get_files(instrument, datafolder, extension, prefix, compression='gzip'):

    """
    Get the file system for instrument

    Parameters
    ----------
    instrument: string
        instrument
    datafolder: basestring
        folder with the data
    extension: string
        file extension of files to analize
    prefix: string 
        Prefix that files must containt to be analyzed
    compression: string
        Compression algorithm used to open files

    Returns
    -------
    a list of dask type files object

    """

    list_of_files = []
    for dirpath, dirnames, filenames in os.walk(top=datafolder, followlinks=True):
        if prefix is not None:
            list_of_files += [os.path.join(dirpath, file) for file in filenames if file.endswith(extension) and prefix in file]
        else:
            list_of_files += [os.path.join(dirpath, file) for file in filenames if file.endswith(extension)]

    if compression is not None:
        return dask.bytes.open_files(list_of_files, compression=compression)
    else:
        return dask.bytes.open_files(list_of_files)


def read_files(instrument, datafolder, destination, indexfile, nworkers=4, extension='.fits', prefix=None, compression=None):
    """
    Read all fits for instrument, get date if opens,
    otherwise record error message

    Parameters
    ----------
    instrument: string
        instrument
    datafolder: string
        folder containing files to analyze
    destination: string
        folder to put the dataframe registering all fits files
    indexfile: string
        Name of index file

    Keywords
    --------
    nworkers: integer
        number of concurrent workers
    extension: string
        file extension of files to analize
    prefix: string 
        Prefix that files must containt to be analyzed
    compression: string
        Compression algorithm used to open files


    Return
    ------

    """

    file_handles = get_files(instrument, datafolder, extension, prefix, compression=compression)

    futures = []

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        for handle in file_handles:
            futures.append(
                executor.submit(extract_date, handle, instrument)
                )

    wait(futures)

    file_information = [f.result() for f in futures]
    data = pd.DataFrame(file_information)

    utils.save_index(data, destination, indexfile)
    return data


def extract_date(handle, instrument):
    """
    Read a fits file and get date and file path

    Parameters
    ----------
    handle: dask handle
    instrument: string

    Returns
    -------
    dictionary with filepath, dateobs, did_not_open flag, instrument,
    error_message (if not opened)
    """

    with handle as file:
        try:
            sun_map = utils.map_prep(file, instrument)
            date = sun_map.date.datetime
            filepath = handle.path

            file_information = {'filepath': filepath,
                                'dateobs': date,
                                'rsun_obs': sun_map.rsun_obs.value,
                                'did_not_open': 0,
                                'instrument': instrument,
                                'error_message': ''
                                }

        except Exception as e:  # In case the file is corrupt / doesn't open
            filepath = handle.path
            file_information = {'filepath': filepath,
                                'dateobs': None,
                                'rsun_obs': None,
                                'did_not_open': 1,
                                'instrument': instrument,
                                'error_message': e
                                }
    return file_information


if __name__ == '__main__':


    args = parser.parse_args()

    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))

    args.nworkers = int(args.nworkers)

    data = read_files(args.instrument, 
                        args.datafolder, 
                        args.destination, 
                        args.indexfile, 
                        nworkers=args.nworkers,
                        extension=args.extension,
                        prefix=args.prefix,
                        compression=args.compression)

    print(data)
