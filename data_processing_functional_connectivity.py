"""
Function for data pre-processing to get dynamic function connectivity
multidimensional matrix.

Katerina Capouskova 2018-2020, kcapouskova@hotmail.com
"""

import os

import numpy as np
import pylab
from numpy.ma import *
from scipy import signal
from scipy.constants import pi
from scipy.signal import butter
from tqdm import tqdm

from utilities import create_dir, find_delimeter


def convert_to_phases(input_path, output_path, brain_areas, t_phases, subject, TR):
    """
    Converts BOLD signal into phases by Hilbert Transform with filtering included.

    :param input_path: path to input file
    :type input_path: str
    :param output_path: path to output directory
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :param t_phases: number of time phases
    :type t_phases: int
    :param subject: subject number
    :type subject: int
    :param TR: repetition time
    :type TR: int
    :return: phases matrix
    :rtype: np.ndarray
    """
    phases = np.full((brain_areas, t_phases), fill_value=0, dtype=np.float64)
    delim = find_delimeter(input_path)
    array = np.genfromtxt(input_path, delimiter=delim)
    for area in tqdm(range(0, brain_areas)):
        # select by columns, transform to phase
        time_series = pylab.demean(signal.detrend(array[:, area]))
        filtered_ts = filter_signal(time_series, TR)
        phases[area, :] = np.angle(signal.hilbert(filtered_ts))
    np.savez_compressed(os.path.join(output_path, 'phases_{}'.format(subject)), phases)
    return phases


def filter_signal(time_series, TR):
    """
    Performs bandpass filtering of BOLD signal data.

    :param time_series: time series array
    :type time_series: np.ndarray
    :param TR: repetition time
    :type TR: int
    :return: filtered time series
    :rtype: np.ndarray
    """
    # Nyquist
    nyq = 1.0 / (2.0 * TR)
    # Lowpass frequency of filter (Hz)
    low = 0.04 / nyq
    # Highpass frequency of filter (Hz)
    high = 0.07 / nyq
    Wn = [low, high]
    # 2nd order butterworth filter
    k = 2
    # Constructing the filter
    b, a = butter(k, Wn, btype='bandpass', output='ba')
    # Filtering
    filt_ts = signal.lfilter(b, a, time_series)
    return filt_ts


def dynamic_functional_connectivity(paths, output_path, brain_areas,
                                    pattern, t_phases, n_subjects, TR):
    """
    Computes the dynamic functional connectivity of brain areas.

    :param paths: list of paths in input dir
    :type paths: []
    :param output_path: path to output directory
    :type output_path: str
    :param brain_areas: number of brain areas
    :type brain_areas: int
    :param pattern: pattern of input files
    :type pattern: str
    :param t_phases: number of time points
    :type t_phases: int
    :param n_subjects: number of subjects
    :type n_subjects:int
    :param TR: repetition time
    :type TR: int
    :return: dFC output path
    :rtype: str
    """
    dFC = np.full((brain_areas, brain_areas), fill_value=0, dtype=np.float64)

    for n in tqdm(range(n_subjects)):
        phases = convert_to_phases(paths[n], output_path, brain_areas, t_phases, n, TR)
        for t in range(0, t_phases):
            for i in range(0, brain_areas):
                for z in range(0, brain_areas):
                    if absolute(phases[i, t] - phases[z, t]) > pi:
                        dFC[i, z] = cos(2 * pi - absolute(
                            phases[i, t] - phases[z, t]))
                    else:
                        dFC[i, z] = cos(absolute(phases[i, t] -
                                                       phases[z, t]))
            dfc_output = os.path.join(output_path, 'dFC')
            create_dir(dfc_output)
            np.savez_compressed(os.path.join(dfc_output, 'subject_{}_time_{}'.format(n, t)), dFC)

    return dfc_output
