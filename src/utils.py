"""
Common low level utility functions
"""

import logging
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from sunpy.map import Map
import astropy.units as u
from astropy.io import fits

def create_hist_bins(dl=0.02, ax_lim=3000, noise_level=0.2):
    """
    Generate bins for 2d histograms
    :param dl:
    :param ax_lim:
    :param noise_level:
    :return:
    """

    lim = np.log10(ax_lim)
    bins = np.round(np.power(10, np.arange(1, lim + dl, dl)), 2)
    bins = bins - 10 + noise_level
    bins = np.append(np.flip(-(bins)), bins)

    return bins


def disable_warnings():
    """
    Disable printing of warnings

    Returns
    -------
    None
    """
    warnings.simplefilter("ignore")


def file_exist(bucketname, filename):

    """
    Check whether a file exists in a bucket

    Parameters
    ----------
    bucketname: string
        name of bucket or file to check file existence

    filename: string
        name of file to check existence

    client:
        google.storage.Client()

    Returns
    -------
    bool: True iff the file exist

    """

    stats = os.path.exists(os.path.join(bucketname, filename))

    return stats


def get_logger(name):
    """
    Return a logger for current module
    Returns
    -------

    logger : logger instance

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                                  datefmt="%Y-%m-%d - %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logfile = logging.FileHandler('run.log', 'w')
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)

    return logger


def map_prep(file, instrument, center_rotate=False, crop=False, chunks=32, transpose=False):
    """
    Return a processed hmi magnetogram and path

    Parameters
    ----------
    file : file desctiptor
        File to open

    instrument: string
        Instrument to process

    center_rotate : Bool
        Whether to rotate and center the file so that the north points upwards
        and the sun is centered on the detector
    
    crop : Bool
        Whether to crop the image to ensure that the size is an integer multiple of chunks

    chunks: int
        number of chunks on each dimension that user wants the image to have

    transpose :  Bool
        Whether to transpose the map data

    Returns
    ----------

    map : sunpy.map
        preped sunpy map
    -------

    """

    # Open fits file as HUDL and fix header
    hdul = fits.open(file, cache=False)
    hdul.verify('fix')

    # Assemble Sunpy map (compressed fits file so use second hdu)

    if len(hdul) == 2:
        header = hdul[1].header
        data = hdul[1].data
        header['DSUN_REF'] = 149597870691

    elif len(hdul) == 1:
        if instrument == 'mdi':
            header = hdul[0].header
            header['RSUN_OBS'] = header['OBS_R0']
            header['RSUN_REF'] = 696000000
            header['CROTA2'] = -header['SOLAR_P0']
            header['CRVAL1'] = 0.000000
            header['CRVAL2'] = 0.000000
            header['CUNIT1'] = 'arcsec'
            header['CUNIT2'] = 'arcsec'
            header['DSUN_OBS'] = header['OBS_DIST']
            header['DSUN_REF'] = 1

            header.pop('SOLAR_P0')
            header.pop('OBS_DIST')
            header.pop('OBS_R0')

            data = hdul[0].data

        if instrument == 'gong':
            ## Update header to make it useful
            header = hdul[0].header
            header['RSUN_OBS'] = (header['RADIUS'] * u.rad).to(u.arcsec).value
            header['RSUN_REF'] = 696000000
            header['CROTA2'] = 0
            header['CUNIT1'] = 'arcsec'
            header['CUNIT2'] = 'arcsec'
            header['DSUN_REF'] = 149597870691
            header['DSUN_OBS'] = header['DSUN_REF'] * header['DISTANCE']
            header['cdelt1'] = 2.5310 + 0.0005
            header['cdelt2'] = 2.5310 + 0.0005
            header['crpix1'] = header['FNDLMBXC'] + 0.5
            header['crpix2'] = header['FNDLMBYC'] + 0.5
            header['R_SUN'] = header['RSUN_OBS'] / header['cdelt2']
            header['CTYPE1'] = 'HPLN-TAN'
            header['CTYPE2'] = 'HPLT-TAN'
            date = header['DATE-OBS']
            header['DATE-OBS'] = date[0:4] + '-' + date[5:7] + '-' + date[8:10] + 'T' + header['TIME-OBS'][0:11] + '0'

            data = hdul[0].data

        if instrument == 'spmg':
            header = hdul[0].header
            header['cunit1'] = 'arcsec'
            header['cunit2'] = 'arcsec'
            header['CDELT1'] = header['CDELT1A']
            header['CDELT2'] = header['CDELT2A']
            header['CRVAL1'] = 0
            header['CRVAL2'] = 0
            header['RSUN_OBS'] = header['EPH_R0 ']
            header['CROTA2'] = 0
            header['CRPIX1'] = header['CRPIX1A']
            header['CRPIX2'] = header['CRPIX2A']
            header['PC2_1'] = 0
            header['PC1_2'] = 0
            header['RSUN_REF'] = 696000000

            # Adding distance to header
            t = Time(header['DATE-OBS'])
            loc = EarthLocation.of_site('kpno')
            with solar_system_ephemeris.set('builtin'):
                sun = get_body('sun', t, loc)
            header['DSUN_OBS'] = sun.distance.to('m').value
            header['DSUN_REF'] = 149597870691

            # selecting right layer for data
            data = hdul[0].data[5, :, :]

        if instrument == 'kp512':
            header = hdul[0].header
            header['cunit1'] = 'arcsec'
            header['cunit2'] = 'arcsec'
            header['CDELT1'] = header['CDELT1A']
            header['CDELT2'] = header['CDELT2A']
            header['CRVAL1'] = 0
            header['CRVAL2'] = 0
            header['RSUN_OBS'] = header['EPH_R0 ']
            header['CROTA2'] = 0
            header['CRPIX1'] = header['CRPIX1A']
            header['CRPIX2'] = header['CRPIX2A']
            header['PC2_1'] = 0
            header['PC1_2'] = 0
            header['RSUN_REF'] = 696000000

            # Adding distance to header
            t = Time(header['DATE-OBS'])
            loc = EarthLocation.of_site('kpno')
            with solar_system_ephemeris.set('builtin'):
                sun = get_body('sun', t, loc)
            header['DSUN_OBS'] = sun.distance.to('m').value
            header['DSUN_REF'] = 149597870691

            # selecting right layer for data
            data = hdul[0].data[2, :, :]

        if instrument == 'mwo':

            file_name = file.name

            # Deconstruct Name to assess date
            tmpPos = file_name.rfind('_')

            year = int(file_name[tmpPos - 6:tmpPos - 4])

            # Adding century
            if year < 1960:
                year += 2000
            else:
                year += 1900

            month = int(file_name[tmpPos - 4:tmpPos - 2])
            day = int(file_name[tmpPos - 2:tmpPos])
            hr = int(file_name[tmpPos + 1:tmpPos + 3]) - 1
            mn = int(file_name[tmpPos + 3:tmpPos + 5])
            sc = 0

            # Fix Times
            if mn > 59:
                mn = mn - 60
                hr = hr + 1

            # Assemble date
            if hr > 23:
                tmpDate = datetime.datetime(year, month, day, hr - 24, mn,
                                            sc) + datetime.timedelta(days=1)
            else:
                tmpDate = datetime.datetime(year, month, day, hr, mn, sc)

            header = hdul[0].header
            header['CUNIT1'] = 'arcsec'
            header['CUNIT2'] = 'arcsec'
            header['CDELT1'] = header['DXB_IMG']
            header['CDELT2'] = header['DYB_IMG']
            header['CRVAL1'] = 0.0
            header['CRVAL2'] = 0.0
            header['RSUN_OBS'] = (header['R0']) * header['DXB_IMG']
            header['CROTA2'] = 0.0
            header['CRPIX1'] = header['X0'] - 0.5
            header['CRPIX2'] = header['Y0'] - 0.5
            header['T_OBS'] = tmpDate.strftime('%Y-%m-%dT%H-%M:00.0')
            header['DATE-OBS'] = tmpDate.strftime('%Y-%m-%dT%H:%M:00.0')
            header['DATE_OBS'] = tmpDate.strftime('%Y-%m-%dT%H:%M:00.0')
            header['RSUN_REF'] = 696000000
            header['CTYPE1'] = 'HPLN-TAN'
            header['CTYPE2'] = 'HPLT-TAN'
            header['RSUN_REF'] = 696000000

            # Adding distance to header
            t = Time(header['DATE-OBS'], format='isot')
            loc = EarthLocation.of_site('mwo')
            with solar_system_ephemeris.set('builtin'):
                sun = get_body('sun', t, loc)
            header['DSUN_OBS'] = sun.distance.to('m').value
            header['DSUN_REF'] = 149597870691

            # selecting right layer for data
            data = hdul[0].data

    if transpose:
        sun_map = Map(data.T, header)
    else:
        sun_map = Map(data, header)
    

    if center_rotate:
        sun_map = sun_map.rotate(recenter=True)

    # Calculate new shape and how much needs to be trimmed
    new_shape = data.shape[0]
    if crop:
        new_shape = int(data.shape[0]//chunks)*chunks

    if center_rotate or crop:
        shape_trim = int((gong_fits[0].data.shape[0]-new_shape)/2)  

        # # Crop image to desired shape
        sz_x_diff = (sun_map.data.shape[0]-data.shape[0])//2 + shape_trim
        sz_y_diff = (sun_map.data.shape[1]-data.shape[1])//2 + shape_trim

        sun_map.meta['crpix1'] = sun_map.meta['crpix1']-sz_x_diff
        sun_map.meta['crpix2'] = sun_map.meta['crpix2']-sz_y_diff
    
        sun_map = Map(sun_map.data[sz_x_diff:sz_x_diff+new_shape, sz_y_diff:sz_y_diff+new_shape].copy(), sun_map.meta)   


    hdul.close()
    return sun_map


def set_random_seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return seed


def save_index(data, destination, filename):

    """
    Upload a pandas dataframe with metrics and an
    index for each file in destination

    Parameters
    ----------
    data: pandas dataframe
        metrics, date and filename
    destination: string
        name of the folder destination to load index file

    Returns
    -------

    """

    if file_exist(destination, '{}.csv'.format(filename)):
        index_df = pd.read_csv('{}/{}.csv'.format(destination, filename),
                               index_col='index')
        index_df['dateobs'] = pd.to_datetime(index_df['dateobs'])
        data = pd.concat([data, index_df]).drop_duplicates('dateobs')

    data.sort_values(by='dateobs', inplace=True)
    data.set_index(np.arange(len(data)), inplace=True)
    data.index.name = 'index'
    data.dropna(inplace=True)

    data.to_csv('{}/{}.csv'.format(destination, filename))