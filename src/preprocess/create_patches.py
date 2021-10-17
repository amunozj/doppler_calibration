import itertools
import argparse
import datetime
import os
import random
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor

import astropy.units as u
import dask.bytes
import numpy as np
import pandas as pd
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from google.cloud import storage
from pandas import Grouper
from skimage.transform import resize
from sklearn.feature_extraction import image

from sunpy.map import Map
from sunpy.physics.differential_rotation import differential_rotate
from skimage.feature import match_template
from scipy.ndimage import fourier_shift, gaussian_filter

_PREPROCESS_DIR = os.path.abspath(__file__).split('/')[:-1]
_SRC_DIR = os.path.join('/',*_PREPROCESS_DIR[:-1])
sys.path.append(_SRC_DIR)

# # Native packages
import utils

logger = utils.get_logger(__name__)


def register(source_map, target_map, patchscale, align=True, buffer=0.5, reproject=False, noise_level=0):
    scale_factor = int(np.power(2, np.round(np.log2(target_map.data.shape[0] / source_map.data.shape[0]))))
    lr_patch_size = int(source_map.data.shape[0] / patchscale[0])
    lr_stride = int(source_map.data.shape[0] / patchscale[1])
    hr_patch_size = lr_patch_size * scale_factor
    hr_stride = lr_stride * scale_factor

    if align:
        # upsample source and smooth target
        source_upsampled = resize(source_map.data,
                                  (source_map.data.shape[0] * scale_factor, source_map.data.shape[1] * scale_factor))

        # Defining new meta parameters
        new_meta = source_map.meta.copy()
        new_meta['crpix1'] = (new_meta['crpix1'] - source_map.data.shape[0] / 2 - 0.5) * scale_factor + \
                             source_map.data.shape[0] * scale_factor / 2 + 0.5
        new_meta['crpix2'] = (new_meta['crpix2'] - source_map.data.shape[1] / 2 - 0.5) * scale_factor + \
                             source_map.data.shape[1] * scale_factor / 2 + 0.5
        new_meta['cdelt1'] = new_meta['cdelt1'] / scale_factor
        new_meta['cdelt2'] = new_meta['cdelt2'] / scale_factor

        source_upsampled = Map(source_upsampled, new_meta)

        # Reproject target into source
        if reproject:
            output, footprint = reproject_interp(target_map, source_upsampled.wcs, source_upsampled.data.shape,
                                                 order='bicubic')
            new_meta = source_upsampled.meta
            target_map = Map(output, source_upsampled.meta)

            x, y = np.meshgrid(*[np.arange(v.value) for v in target_map.dimensions])*u.pixel
            hpc_coords = target_map.pixel_to_world(x, y)
            rSun = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / target_map.rsun_obs
            target_map.data[rSun > 1] = 0
            target_map.data[np.isnan(target_map.data)] = 0

        target_smooth = gaussian_filter(target_map.data, scale_factor / 1.5)
        target_smooth = Map(target_smooth, target_map.meta)

        # create patches
        target_smooth_patch = get_patch(target_smooth, hr_patch_size, stride=hr_stride)
        target_patch = get_patch(target_map, hr_patch_size, stride=hr_stride)
        source_patch = get_patch(source_map, lr_patch_size, stride=lr_stride)
        source_upsampled_patch = get_patch(source_upsampled, hr_patch_size, stride=hr_stride)

        target_shifted = target_patch[:, 0:2, :, :].copy()

        # Padded target map
        target_map_padded = np.pad(target_map.data, int(hr_patch_size / buffer), mode='constant', constant_values=0)

        # Fixing header
        new_meta = target_map.meta.copy()
        new_meta['crpix1'] = new_meta['crpix1'] - target_map.data.shape[0] / 2 + target_map_padded.shape[0] / 2
        new_meta['crpix2'] = new_meta['crpix2'] - target_map.data.shape[1] / 2 + target_map_padded.shape[1] / 2

        target_map_padded = Map(target_map_padded, new_meta)
        target_map_padded_patch = get_patch(target_map_padded, int(hr_patch_size + 2 * hr_patch_size / buffer),
                                            stride=hr_stride)

        # Padded target smoothed map
        target_smooth_padded = np.pad(target_smooth.data, int(hr_patch_size / buffer), mode='constant', constant_values=0)
        target_smooth_padded = Map(target_smooth_padded, new_meta)
        target_smooth_padded_patch = get_patch(target_smooth_padded, int(hr_patch_size + 2 * hr_patch_size / buffer),
                                               stride=hr_stride)

        # Variable to store shifts for the distortion map
        shifts = np.zeros((target_patch.shape[0], 2))

        # Assemble the right patches using template matching
        for i in range(target_smooth_patch.shape[0]):

            # Make sure there is a pattern to match above the noise level
            if np.sum(np.abs(source_patch[i, 0, :, :]) > noise_level) / (
                    source_patch.shape[2] * source_patch.shape[3]) < 0.04:
                source_patch[i, 0, :, :] = source_patch[i, 0, :, :] * 0
                source_upsampled_patch[i, 0, :, :] = source_upsampled_patch[i, 0, :, :] * 0

            if np.sum(np.abs(target_patch[i, 0, :, :]) > noise_level) / (
                    target_patch.shape[2] * target_patch.shape[3]) < 0.04:
                target_patch[i, 0, :, :] = target_patch[i, 0, :, :] * 0
                target_smooth_patch[i, 0, :, :] = target_smooth_patch[i, 0, :, :] * 0
                target_smooth_padded_patch[i, 0, :, :] = target_smooth_padded_patch[i, 0, :, :] * 0
                target_map_padded_patch[i, 0, :, :] = target_map_padded_patch[i, 0, :, :] * 0

            if np.sum(target_smooth_padded_patch[i, 0, :, :]) == 0 or np.sum(source_upsampled_patch[i, 0, :, :]) == 0:
                x = int(hr_patch_size / buffer)
                y = int(hr_patch_size / buffer)
            else:
                # Matching template and finding indices of bottom right corner in extended patch
                tmp_source = source_upsampled_patch[i, 0, :, :].copy()
                tmp_source[np.abs(tmp_source) < noise_level] = 0
                tmp_target = target_smooth_padded_patch[i, 0, :, :].copy()
                tmp_target[np.abs(tmp_target) < noise_level] = 0
                result = match_template(tmp_target, tmp_source)
                ij = np.unravel_index(np.argmax(result), result.shape)
                x, y = ij[::-1]

            target_shifted[i, 0, :, :] = target_map_padded_patch[i, 0, y:y + hr_patch_size,
                                         x:x + hr_patch_size]

            shifts[i, 0] = x - int(hr_patch_size / buffer)
            shifts[i, 1] = y - int(hr_patch_size / buffer)

        return source_patch, target_shifted, shifts

    else:
        source_patch = get_patch(source_map, hr_patch_size, stride=hr_stride)
        target_patch = get_patch(target_map, hr_patch_size, stride=hr_stride)

        return source_patch, target_patch, []


def get_files_from_index(index_df, source, target):
    """
    Get the file system for any instrument using the data_index.csv

    Parameters
    ----------
    index_df: pandas.DataFrame()
    source: str
        instrument source
    target: str
        instrument target

    Returns
    -------
    a list of tuple with dask type files object and instrument name

    """
    source_list = list(index_df['filepath_{}'.format(source)])
    target_list = list(index_df['filepath_{}'.format(target)])

    source_tuples = zip([source] * len(source_list), dask.bytes.open_files(source_list))
    target_tuples = zip([target] * len(target_list), dask.bytes.open_files(target_list))

    return zip(target_tuples, source_tuples, )


def upload_patches(patches_list):
    """
    Requires: tuple in patches_list are (path_source, path_target, source, target)
    with source and target are numpy array and path_source, path_target are strings

    Effect: takes each tuple in patch list and save them if they are not
    zeros.

    :param patches_list: list
    :return: list of tuple (path_source + patch_num, path_target  patch_num, patch_num)
    """

    patch_num = 0
    file_info = []

    for path_source, path_target, source, target in patches_list:

        patch_num += 1

        # remove arrays with all zeros for source and target
        if np.all(source == 0) & np.all(target == 0):
            continue

        path_source = '{}_{}'.format(path_source, patch_num)
        path_target = '{}_{}'.format(path_target, patch_num)

        np.save(path_source, source)
        np.save(path_target, target)

        file_info.append((path_source, path_target, patch_num))

    return file_info


def run_process(files_handle, target, patchscale, destination, noise_level=0, only_target=False,
                diff_rotate=True, register_trans=False, reproject=False, downscale=None, cropped=False, phase='train'):
    """
    Requires: file_handles is a list of fits file with the target in the first
    position
    Effect: from a full disk source and target, get the corresponding source and
    target patches and save them in the destination folder

    :param files_handle: dask list
    :param target: string
        name of target instrument
    :param patchscale: tuple
        size, stride
    :param destination: string

    :param diff_rotate: bool
        Only store target patches in case of same input and target

    :param reproject: bool
        Assemmble patche directly from raw magnetograms

    :param downscale: int
        downscale factor

    :param cropped: bool
        whethter to crop at the center

    :return:
    """

    file_info = []

    shifts_folder = '{}/shifts'.format(destination)
    os.makedirs(shifts_folder, exist_ok=True)

    for instrument, handle in files_handle:
        with handle as file:
            logger.info(handle.path)
            try:
                amap = map_prep(file, instrument)

                # Zeroing Nans and pixels outside the solar radius
                x, y = np.meshgrid(*[np.arange(v.value) for v in amap.dimensions]) * u.pixel
                hpc_coords = amap.pixel_to_world(x, y)
                rSun = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / amap.rsun_obs
                amap.data[rSun > 1] = 0
                amap.data[np.isnan(amap.data)] = 0

                date = amap.date.datetime
                year = date.year
                month = date.month
                day = date.day

                if instrument == target:  # need to get the position of the observer
                    observer = amap.observer_coordinate
                    target_datetime = amap.date.to_datetime()

                elif diff_rotate:  # diff rotate the source
                    source_datetime = amap.date.to_datetime()
                    amap = differential_rotate(amap, observer=observer)

                elif downscale is not None:
                    source_datetime = amap.date.to_datetime()
                    amap = downsample(amap, downscale)

                else:
                    source_datetime = amap.date.to_datetime()

                os.makedirs('{}/{}/{}/{}/{}'.format(destination, phase, year, month, day), exist_ok=True)
                destination_patch = '{}/{}/{}'.format(year, month, day)
                patch_path = '{}/{}_{}'.format(destination_patch, instrument, date.strftime("%Y%m%d-%H%M%S"))

                if instrument == target:  # record target
                    target_path = patch_path
                    target_map = amap

                elif cropped:
                    source_cropped = crop(amap, patchscale[0])
                    target_cropped = crop(target_map, patchscale[0])

                    save_path = '{}/{}/{}.npy'.format(destination, phase, patch_path)
                    np.save(save_path, source_cropped.data)
                    save_path = '{}/{}/{}.npy'.format(destination, phase, target_path)
                    np.save(save_path, target_cropped.data)

                    file_info.append(('{}.npy'.format(patch_path),
                                      '{}.npy'.format(target_path), date,
                                      (target_datetime - source_datetime).total_seconds()))

                else:  # shift and save patches
                    source_patch, target_patch, shifts = register(amap, target_map, patchscale, align=register_trans,
                                                                  reproject=reproject, noise_level=noise_level)

                    shifts_path = '{}/{}_{}.npy'.format(shifts_folder, instrument, date.strftime("%Y%m%d-%H%M%S"))
                    np.save(shifts_path, shifts)

                    for i in np.arange(source_patch.shape[0]):
                        if np.all(target_patch[i, 0, :, :] == 0) or np.all(source_patch[i, 0, :, :] == 0):
                            continue
                        if np.sum(target_patch[i, 0, :, :]) == 0 or np.sum(source_patch[i, 0, :, :]) == 0:
                            continue

                        if not only_target:
                            save_path = '{}/{}/{}_{}.npy'.format(destination, phase, patch_path, i)
                            np.save(save_path, source_patch[i, :, :, :])
                        save_path = '{}/{}/{}_{}.npy'.format(destination, phase, target_path, i)
                        np.save(save_path, target_patch[i, :, :, :])

                        file_info.append(('{}_{}.npy'.format(patch_path, i),
                                          '{}_{}.npy'.format(target_path, i),
                                          i, date,
                                          (target_datetime - source_datetime).total_seconds()))

            except Exception as e:
                logger.info(f'Error in file {handle.path} with error: {e}')
                file_info = []

    return file_info


def export(source, target, destination, source_data, target_data, tolerance, patchscale, diff_rotate=True,
           register_trans=False, cropped=False, downscale=None, nworkers=1, sampling=None, only_times=False,
           only_target=False, reproject=False, noise_level=0):
    index_filepath1 = "{}/index.csv".format(source_data)
    index_df1 = pd.read_csv(index_filepath1)

    index_filepath2 = "{}/index.csv".format(target_data)
    index_df2 = pd.read_csv(index_filepath2)

    if (source == target) & (downscale is not None):
        source = 'LR{}_'.format(downscale) + source
        target = 'HR_' + target

    index_df1['instrument'] = source
    index_df2['instrument'] = target

    index_df = pd.concat([index_df1, index_df2], axis=0)
    index_df.dropna(subset=['dateobs'], inplace=True)

    index_df['dateobs'] = pd.to_datetime(index_df['dateobs'])

    # time alignment
    inst1_inst2_index = time_alignment(index_df, source, target, tolerance)
    print(inst1_inst2_index)

    if only_times:
        os.makedirs('{}'.format(destination), exist_ok=True)
        upload_index(inst1_inst2_index, '{}'.format(destination), 'index')

    else:

        if sampling is not None:
            index = np.random.choice(inst1_inst2_index.index, int(len(inst1_inst2_index) / sampling), replace=False)
            inst1_inst2_index = inst1_inst2_index.loc[index, :]

        # train / test / validate split
        train_test_val_dict = train_test_split(inst1_inst2_index)

        # files handles corresponding to aligned index for train, test, validate
        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            # with ThreadPoolExecutor(max_workers=nworkers) as executor:
            for type_df, df_time_list in train_test_val_dict.items():
                d_list = [df_time[1] for df_time in df_time_list]
                df = pd.concat(d_list)

                file_handles = get_files_from_index(df, source, target)
                futures = []

                for file_handle in file_handles:
                    futures.append(executor.submit(run_process,
                                                   file_handle,
                                                   target,
                                                   patchscale,
                                                   destination,
                                                   noise_level=noise_level,
                                                   diff_rotate=diff_rotate,
                                                   reproject=reproject,
                                                   only_target=only_target,
                                                   register_trans=register_trans,
                                                   downscale=downscale,
                                                   cropped=cropped,
                                                   phase=type_df))

                wait(futures)
                results = [f.result() for f in futures]
                results_flat = itertools.chain.from_iterable(results)

                if cropped:
                    index_exported_files = pd.DataFrame(results_flat,
                                                        columns=['filename_source', 'filename_target', 'dateobs',
                                                                 'timedelta'])
                else:
                    index_exported_files = pd.DataFrame(results_flat,
                                                        columns=['filename_source', 'filename_target', 'patch_num',
                                                                 'dateobs', 'timedelta'])

                upload_index(index_exported_files, '{}/{}'.format(destination, type_df), 'index')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--destination')
    parser.add_argument('--source')
    parser.add_argument('--target')
    parser.add_argument('--source_data')
    parser.add_argument('--target_data')
    parser.add_argument('--patchscale', nargs='+', type=float)
    parser.add_argument('--tolerance', type=float)
    parser.add_argument('--nworkers')
    parser.add_argument('--sampling')
    parser.add_argument('--noise_level', type=float)
    parser.add_argument('--downscale', type=float)
    parser.add_argument('--cropped', action='store_true')
    parser.add_argument('--diff_rotate', action='store_true')
    parser.add_argument('--register_trans', action='store_true')
    parser.add_argument('--only_times', action='store_true')
    parser.add_argument('--only_target', action='store_true')
    parser.add_argument('--reproject', action='store_true')

    args = parser.parse_args()

    sampling = None
    if args.sampling is not None:
        sampling = int(args.sampling)

    noise_level = 0
    if args.noise_level is not None:
        noise_level = args.noise_level

    kwargs_dict = {'nworkers': int(args.nworkers), 'sampling': sampling}
    # 'downscale': args.downscale}

    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))

    export(args.source,
           args.target,
           args.destination,
           args.source_data,
           args.target_data,
           args.tolerance,
           args.patchscale,
           args.diff_rotate,
           args.register_trans,
           args.cropped,
           noise_level=noise_level,
           reproject=args.reproject,
           only_times=args.only_times,
           only_target=args.only_target,
           downscale=args.downscale,
           nworkers=int(args.nworkers),
           sampling=sampling)
