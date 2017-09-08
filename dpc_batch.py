'''
Created on Feb 23, 2017

@author: Mirna Lerotic, 2nd Look
'''
from __future__ import (print_function, division)
import sys
import os
import numpy as np
import multiprocessing as mp
import h5py
import PIL
try:
    from tifffile import imsave
    havetiff = True
except ImportError as ex:
    print('[!] Import error - tifffile not available. Tif files will not be saved')
    print('[!] (import error: {})'.format(ex))
    havetiff = False

# try:
#     from databroker import db, get_events
# except ImportError as ex:
#     print('[!] Unable to import DataBroker library.')
from hxn_db_config import db

try:
    import hxntools
    import hxntools.handlers
    from hxntools.scan_info import ScanInfo
    #from databroker import DataBroker
except ImportError as ex:
    print('[!] Unable to import hxntools library.')
    print('[!] (import error: {})'.format(ex))
    hxntools = None
else:
    hxntools.handlers.register()


import dpc_kernel as dpc

version = '0.1.0'


def load_scan_from_mds(scan_id):
    #hdrs = DataBroker(scan_id=int(scan_id))
    hdrs = db[int(scan_id)]
    if len(hdrs) == 1:
        hdr = hdrs[0]

    return ScanInfo(hdr)


def get_ref_from_mds(scan, first_image, file_store_key):
        if scan is None:
            return

        iter_ = iter(scan)

        first_image = max((1, first_image + 1))
        ref_image = None

        try:
            for i in range(first_image):
                ref_image = next(iter_)
        except StopIteration:
            print('Reference image #{} does not exist with data key {}'
                  ''.format(first_image, file_store_key))

        print(ref_image)

        return ref_image


def set_scan_from_scaninfo(scan):

    if scan.dimensions is None or len(scan.dimensions) == 0:
        return

    scan_range = scan.range
    print('Scan dimensions', scan.dimensions)
    print('Scan range:', scan_range)

    pyramid_scan = scan.pyramid

    if isinstance(scan_range, dict):
        scan_range = [scan_range[mtr] for mtr in scan.motors]

    if len(scan.dimensions) == 1:
        nx, ny = scan.dimensions[0], 1
        if scan_range is not None:
            dx = np.diff(scan_range[0]) / nx
            dy = 0.0
    else:
        nx, ny = scan.dimensions
        if scan_range is not None:
            dx = np.diff(scan_range[0]) / nx
            dy = np.diff(scan_range[1]) / ny

    cols = nx
    rows = ny

    return dx, dy, cols, rows, pyramid_scan


def load_data_hdf5(path):
    """
    Read images using the h5py lib
    """
    f = h5py.File(str(path), 'r')
    entry = f['entry']
    instrument = entry['instrument']
    detector = instrument['detector']
    dsdata = detector['data']
    data = dsdata[...]

    return np.array(data)


def load_image_hdf5(path):

    data = load_data_hdf5(path)

    return data[0, :, :]


def save_results(a, gx, gy, phi, rx, ry, save_path, save_filename, scan_number, save_pngs=True, save_tif=True, save_txt=True):

    save_filename = os.path.join(save_path, 'S{0}_{1}'.format(scan_number, save_filename))

    if os.path.isdir(save_path):
        if save_txt:
            a_path = save_filename + '_a.txt'
            np.savetxt(a_path, a)
            gx_path = save_filename + '_gx.txt'
            np.savetxt(gx_path, gx)
            gy_path = save_filename + '_gy.txt'
            np.savetxt(gy_path, gy)
            rx_path = save_filename + '_rx.txt'
            np.savetxt(rx_path, rx)
            ry_path = save_filename + '_ry.txt'
            np.savetxt(ry_path, ry)
            if phi is not None:
                phi_path = save_filename + '_phi.txt'
                np.savetxt(phi_path, phi)

        if save_pngs:
            a_path = save_filename + '_a.png'
            im = PIL.Image.fromarray((2.0 / a.ptp() * (a - a.min())).astype(np.uint8))
            im.save(a_path)
            gx_path = save_filename + '_gx.png'
            im = PIL.Image.fromarray((255.0 / gx.ptp() * (gx - gx.min())).astype(np.uint8))
            im.save(gx_path)
            gy_path = save_filename + '_gy.png'
            im = PIL.Image.fromarray((255.0 / gy.ptp() * (gy - gy.min())).astype(np.uint8))
            im.save(gy_path)
            rx_path = save_filename + '_rx.png'
            im = PIL.Image.fromarray((255.0 / rx.ptp() * (rx - rx.min())).astype(np.uint8))
            im.save(rx_path)
            ry_path = save_filename + '_ry.png'
            im = PIL.Image.fromarray((255.0 / ry.ptp() * (ry - ry.min())).astype(np.uint8))
            im.save(ry_path)
            if phi is not None:
                phi_path = save_filename + '_phi.png'
                im = PIL.Image.fromarray((255.0 / phi.ptp() * (phi - phi.min())).astype(np.uint8))
                im.save(phi_path)

        if save_tif and havetiff:
            if phi is not None:
                imgs = np.stack((a, gx, gy, rx, ry, phi))
                imsave(save_filename + '.tif', imgs.astype(np.float32))
            else:
                imgs = np.stack((a, gx, gy, rx, ry))
                imsave(save_filename + '.tif', imgs.astype(np.float32))

            # a_path = save_filename + '_a.tif'
            # imsave(a_path, a.astype(np.float32))
            # gx_path = save_filename + '_gx.tif'
            # imsave(gx_path, gx.astype(np.float32))
            # gy_path = save_filename + '_gy.tif'
            # imsave(gy_path, gy.astype(np.float32))
            # rx_path = save_filename + '_rx.tif'
            # imsave(rx_path, rx.astype(np.float32))
            # ry_path = save_filename + '_ry.tif'
            # imsave(ry_path, ry.astype(np.float32))
            # if phi is not None:
            #     phi_path = save_filename + '_phi.tif'
            #     imsave(phi_path, phi.astype(np.float32))

    else:
        print('Could not save results! Save directory {0} does not exist.'.format(save_path))

#----------------------------------------------------------------------
def init_scan_parameters():

    #Init settings
    scan_parameters = {
        'file_format' : 'S%d.h5',
        'dx' : 0.1,
        'dy' : 0.1,
        'ref_image' : None,
        'rows' : 121,
        'cols' : 121,
        'start_point' : [1, 0],
        'pixel_size' : 55,
        'focus_to_det' : 1.46,
        'energy' : 19.5,
        'pool' : None,
        'first_image' : 0,
        'roi_x1' : None,
        'roi_x2' : None,
        'roi_y1' : None,
        'roi_y2' : None,
        'bad_pixels' : [],
        'solver' : 'Nelder-Mead',
        'display_fcn' : None,
        'random' : 1,
        'pyramid' : -1,
        'hang' : 1,
        'swap' : -1,
        'reverse_x' : 1,
        'reverse_y' : 1,
        'mosaic_x' : 1,
        'mosaic_y' : 1,
        'load_image' : load_image_hdf5,
        'use_mds' : False,
        'scan' : None,
        'save_path' : None,
        'pad' : False,
    }

    return scan_parameters


#----------------------------------------------------------------------
def read_scan_parameters_from_file(scan_parameters, param_filename):

    print('Reading scan parameters from ', param_filename)

    #try:
    if True:
        f = open(param_filename, 'rt')
        for line in f:

            if line.startswith('#'):
                continue

            elif 'step_size_dx_um' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['dx'] = float(slist[1])

            elif 'step_size_dy_um' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['dy'] = float(slist[1])

            elif 'cols_x' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['cols'] = int(slist[1])

            elif 'rows_y' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['rows'] = int(slist[1])

            elif 'pixel_size_um' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['pixel_size_um'] = float(slist[1])

            elif 'detector_sample_distance' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['focus_to_det'] = float(slist[1])

            elif 'energy_kev ' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['energy']  = float(slist[1])

            elif 'roi_x1' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['roi_x1'] = int(slist[1])

            elif 'roi_x2' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['roi_x2'] = int(slist[1])

            elif 'roi_y1' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['roi_y1'] = int(slist[1])

            elif 'roi_y2' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['roi_y2'] = int(slist[1])

            elif 'mosaic_column_number_x' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['mosaic_x'] = int(slist[1])

            elif 'mosaic_column_number_y' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['mosaic_y'] = int(slist[1])

            elif 'solver' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['solver'] = slist[1].strip()

            elif 'random' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['random'] = int(slist[1])

            elif 'pyramid' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['pyramid'] = int(slist[1])

            elif 'hang' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['hang'] = int(slist[1])

            elif 'swap' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['swap'] = int(slist[1])

            elif 'reverse_x' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['reverse_x'] = int(slist[1])

            elif 'reverse_y' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['reverse_y'] = int(slist[1])

            elif 'pad' in line.lower():
                slist = line.strip().split('=')
                scan_parameters['pad'] = int(slist[1])

        f.close()

        #     except:
        #         print('Could not read the script file. Exiting.')
        #         return

    # for key,value in scan_parameters.items():
    #     print(key + " = " + str(value))

    return scan_parameters


#----------------------------------------------------------------------
def read_scan_parameters_from_datastore(scan_parameters):

    print('Reading scan parameters from DataStore')

    return scan_parameters


#----------------------------------------------------------------------
def read_scan_parameters(scan_parameters, param_filename = '', read_from_datastore = False):

    if read_from_datastore:
        scan_parameters = read_scan_parameters_from_datastore(scan_parameters)
    else:
        scan_parameters = read_scan_parameters_from_file(scan_parameters, param_filename)

    return scan_parameters


#----------------------------------------------------------------------
def parse_scan_range(scan_range, scan_numbers, str_scan_range):

    slist = str_scan_range.split(',')
    for item in slist:
        if '-' in item:
            slist = item.split('-')
            scan_range.append((int(slist[0].strip()), int(slist[1].strip())))
        else:
            scan_numbers.append(int(item.strip()))

    return scan_range, scan_numbers


#----------------------------------------------------------------------
def parse_script(script_file):

    scan_range = []
    scan_numbers = []
    every_nth_scan = 1
    get_data_from_datastore = 0
    scan_header_index = 0
    data_directory = ''
    read_params_from_datastore = 0
    parameter_file = ''
    processes = 1
    scan_header_index = 0
    file_format = 'S{0}.h5'
    save_filename = 'results'
    file_store_key = ''
    save_path = ''
    save_pngs = 1
    save_txt = 1


    try:
    #if True:
        f = open(script_file, 'rt')
        for line in f:

            if line.startswith('#'):
                continue

            elif 'scan_range' in line.lower():
                slist = line.strip().split('=')
                scan_range, scan_numbers = parse_scan_range(scan_range, scan_numbers, slist[1])

            elif 'scan_numbers' in line.lower():
                slist = line.strip().split('=')
                slist = slist[1].split(',')
                for item in slist:
                    scan_numbers.append(int(item.strip()))

            elif 'every_nth_scan' in line.lower():
                slist = line.strip().split('=')
                every_nth_scan = int(slist[1])

            elif 'get_data_from_datastore' in line.lower():
                slist = line.strip().split('=')
                get_data_from_datastore = int(slist[1])

            elif 'read_params_from_datastore' in line.lower():
                slist = line.strip().split('=')
                read_params_from_datastore = int(slist[1])

            elif 'processes' in line.lower():
                slist = line.strip().split('=')
                processes = int(slist[1])

            elif 'scan_header_index' in line.lower():
                slist = line.strip().split('=')
                scan_header_index = int(slist[1])

            elif 'data_directory' in line.lower():
                slist = line.strip().split('=')
                data_directory = slist[1].strip()

            elif 'save_path' in line.lower():
                slist = line.strip().split('=')
                save_path = slist[1].strip()

            elif 'save_filename' in line.lower():
                slist = line.strip().split('=')
                save_filename = slist[1].strip()

            elif 'save_pngs' in line.lower():
                slist = line.strip().split('=')
                save_pngs = int(slist[1])

            elif 'save_txt' in line.lower():
                slist = line.strip().split('=')
                save_txt = int(slist[1])

            elif 'parameter_file' in line.lower():
                slist = line.strip().split('=')
                parameter_file = slist[1].strip()

            elif 'file_format' in line.lower():
                slist = line.strip().split('=')
                file_format = slist[1].strip()

            elif 'file_store_key' in line.lower():
                slist = line.strip().split('=')
                file_store_key = slist[1].strip()

        f.close()

    except:
        print('Could not read the script file. Exiting.')
        exit()

    print('Script setup:')
    print('scan_range', scan_range)
    print('scan_numbers', scan_numbers)
    print('scan_header_index', scan_header_index)
    print('file_store_key', file_store_key)
    print('every_nth_scan', every_nth_scan)
    print('get_data_from_datastore', get_data_from_datastore)
    print('data_directory', data_directory)
    print('file_format', file_format)
    print('read_params_from_datastore', read_params_from_datastore)
    print('parameter_file', parameter_file)
    print('processes', processes)
    print('save_path', save_path)
    print('save_filename', save_filename)
    print('save_pngs', save_pngs)
    print('save_txt', save_txt)

    return scan_range, scan_numbers, every_nth_scan, get_data_from_datastore, data_directory, \
        read_params_from_datastore, parameter_file, processes, scan_header_index, file_format, \
        file_store_key, save_path, save_filename, save_pngs, save_txt


""" ------------------------------------------------------------------------------------------------"""
def run_batch(script_file):

    print('Parsing script ', script_file)
    scan_range, scan_numbers, every_nth_scan, get_data_from_datastore, data_directory, \
        read_params_from_datastore, parameter_file, processes, scan_header_index, file_format, \
        file_store_key, save_path, save_filename, save_pngs, save_txt = parse_script(script_file)

    if get_data_from_datastore == 1:
        if hxntools is None:
            print('Warning! Cannot read scan parameters from DataStore because hnxtools library is not available.')
        print('Reading data from DataStore.')
    else:
        print('Reading data from .h5 files.')

    calc_scan_numbers = np.array((), dtype=np.int)
    calc_scan_numbers = np.append(calc_scan_numbers, np.array(scan_numbers, dtype=np.int))
    #Get scan numbers from the range
    for item in scan_range:
        calc_scan_numbers = np.append(calc_scan_numbers, np.arange(item[0], item[1]+1, every_nth_scan, dtype=np.int))

    scan_parameters = init_scan_parameters()

    try:
        scan_parameters = read_scan_parameters(scan_parameters, param_filename=parameter_file)
    except:
        print('Could not read scan parameters from parameter file {}. Using defaults.'.format(parameter_file))

    dpc_settings = {
        'file_format': '',
        'save_path': data_directory,
        'dx': scan_parameters['dx'],
        'dy': scan_parameters['dy'],
        'x1': scan_parameters['roi_x1'],
        'y1': scan_parameters['roi_y1'],
        'x2': scan_parameters['roi_x2'],
        'y2': scan_parameters['roi_y2'],

        'pixel_size': scan_parameters['pixel_size'],
        'focus_to_det': scan_parameters['focus_to_det'],
        'energy': scan_parameters['energy'],

        'rows': scan_parameters['rows'],
        'cols': scan_parameters['cols'],
        'mosaic_y': scan_parameters['mosaic_y'],
        'mosaic_x': scan_parameters['mosaic_x'],

        'swap': scan_parameters['swap'],
        'reverse_x': scan_parameters['reverse_x'],
        'reverse_y': scan_parameters['reverse_y'],
        'random': scan_parameters['random'],
        'pyramid': scan_parameters['pyramid'],
        'pad': scan_parameters['pad'],
        'hang': scan_parameters['hang'],
        'ref_image': scan_parameters['ref_image'],
        'first_image': scan_parameters['first_image'],
        'solver': scan_parameters['solver'],
        'scan': None,
        'use_mds': scan_parameters['use_mds'],
        'calculate_results': True
    }

    n_scans = calc_scan_numbers.size

    for i_scan in range(n_scans):

        scan_filename = os.path.join(data_directory,file_format.format(calc_scan_numbers[i_scan]))
        print('\nProcessing scan number ', calc_scan_numbers[i_scan])

        dpc_settings['file_format'] = scan_filename
        dpc_settings['ref_image'] = scan_filename
        dpc_settings['scan'] = calc_scan_numbers[i_scan]

        if get_data_from_datastore:
            load_image = dpc.load_image_filestore
            dpc_settings['file_format'] = ''
            dpc_settings['ref_image'] = ''
            dpc_settings['use_hdf5'] = False
            dpc_settings['use_mds'] = True

            try:
                mds_scan = load_scan_from_mds(calc_scan_numbers[i_scan])
            except Exception as ex:
                print('Filestore load failed (datum={}): ({}) {}'
                      ''.format(calc_scan_numbers[i_scan], ex.__class__.__name__, ex))
                raise
            mds_scan.key = file_store_key

            dpc_settings['scan'] = mds_scan

            dpc_settings['ref_image'] = get_ref_from_mds(mds_scan, scan_parameters['first_image'], file_store_key)

            if read_params_from_datastore == 1:
                dx, dy, cols, rows, pyramid_scan = set_scan_from_scaninfo(mds_scan)
                dpc_settings['dx'] = dx
                dpc_settings['dy'] = dy
                dpc_settings['rows'] = rows
                dpc_settings['cols'] = cols
                dpc_settings['pyramid'] = pyramid_scan

        else:
            print('\nProcessing scan ', scan_filename)
            load_image = load_image_hdf5
            dpc_settings['use_hdf5'] = True

        if processes == 0:
            print('Error - number of processes in myscript.txt is equal to 0. Please set to minimum 1 with processes = 1.')
            exit()
        else:
            pool = mp.Pool(processes=processes)


        #Run the analysis
        a, gx, gy, phi, rx, ry = dpc.main(
            pool=pool,
            display_fcn=None,
            load_image=load_image,
            **dpc_settings
        )

        save_results(a, gx, gy, phi, rx, ry, save_path, save_filename,
                     calc_scan_numbers[i_scan],
                     save_pngs=save_pngs, save_tif=True, save_txt=save_txt)

    print('DPC finished')


""" ------------------------------------------------------------------------------------------------"""
def main():

    try:
        script_file = sys.argv[1]
    except:
        print('Error - Script file not given.\nUsage: python dpc_batch.py myscript.txt')
        exit()

    run_batch(script_file)



if __name__ == '__main__':
    main()
