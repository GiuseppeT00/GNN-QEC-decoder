import sys
import os
import time
import json
import pandas as pd
from multiprocessing import Pool, cpu_count, current_process

from src.graph_representation import get_3D_graph

import torch
from torch_geometric.data import Data

import stim
import numpy as np


def syndrome_mask(code_size, repetitions):
    M = code_size + 1
    syndrome_matrix_X = np.zeros((M, M), dtype=np.uint8)
    syndrome_matrix_X[::2, 1:M - 1:2] = 1
    syndrome_matrix_X[1::2, 2::2] = 1
    syndrome_matrix_Z = np.rot90(syndrome_matrix_X) * 3
    syndrome_matrix = (syndrome_matrix_X + syndrome_matrix_Z)
    return np.dstack([syndrome_matrix] * (repetitions + 1))

def stim_to_syndrome_3D(mask, coordinates, stim_data):
    '''
    Converts a stim detection event array to a syndrome grid.
    1 indicates a violated X-stabilizer, 3 a violated Z stabilizer.
    Only the difference between two subsequent cycles is stored.
    '''
    # initialize grid:
    syndrome_3D = np.zeros_like(mask)

    # first to last time-step:
    syndrome_3D[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]] = stim_data

    # only store the difference in two subsequent syndromes:
    syndrome_3D[:, :, 1:] = (syndrome_3D[:, :, 1:] - syndrome_3D[:, :, 0: - 1]) % 2

    # convert X (Z) stabilizers to 1(3) entries in the matrix
    syndrome_3D[np.nonzero(syndrome_3D)] = mask[np.nonzero(syndrome_3D)]

    return syndrome_3D

def generate_batch(stim_data_list,observable_flips_list, detector_coordinates, mask, m_nearest_nodes, power):
    batch = []
    start = time.time()
    for i in range(len(stim_data_list)):
        # convert to syndrome grid:
        syndrome = stim_to_syndrome_3D(mask, detector_coordinates, stim_data_list[i])
        # get the logical equivalence class:
        true_eq_class = np.array([int(observable_flips_list[i])])
        # map to graph representation
        graph = get_3D_graph(syndrome_3D=syndrome,
                             target=true_eq_class,
                             power=power,
                             m_nearest_nodes=m_nearest_nodes)
        batch.append(graph)
    end = time.time()
    print(f'*** Process {current_process().name} : Converted {len(stim_data_list)} stim to graphs in {end - start}s', flush=True)
    return batch


# get the surface code grid:
mask = syndrome_mask(5, 5)
m_nearest_nodes, power = 6, 2
factor = 50
batch_size = 1000
buffer_size = 1000
error_rate = [0.001, 0.002, 0.003, 0.004, 0.005]
stim_data_list = []
observable_flips_list = []

'''
3: 100
5: 600
7: 700
9: 800
11: 900
13: 1000
15: 1100
'''

if __name__ == '__main__':

    #start = time.time()
    #df = pd.read_json('data/circuit_level_noise/d5/d5_dt5_validation_set.json', orient='split')
    #print(f"elapsed for json dump: {time.time() - start}s", flush=True)

    start = time.time()
    df = pd.read_json('prova_js_records.json', orient='records', lines=True, chunksize=50000)
    df = pd.concat(df, ignore_index=True)
    print(df)
    print(df.columns)
    print(f"elapsed for json dump: {time.time() - start}s", flush=True)
    exit(0)

    #df.to_csv('prova_csv.csv', index=False)
    orients = ['split', 'records']

    for o in orients:
        start = time.time()
        df.to_json(f'prova_js_{o}.json', index=False if o == 'split' else True,
                   orient=o, lines=True if o == 'records' else False)
        print(f"elapsed writing for {o}: {time.time() - start}s", flush=True)
    orients = ['split', 'records']
    for o in orients:
        start = time.time()
        if o == 'split':
            df = pd.read_json(f'prova_js_{o}.json', orient=o)
        else:
            df = pd.read_json('prova_js_records.json', orient='records', lines=True, chunksize=10000)
        print(f"elapsed reading for {o}: {time.time() - start}s", flush=True)
    exit(0)

    circuits = []
    for p in error_rate:
        circuits.append(stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=5,
            distance=5,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p))
    detector_coordinates = circuits[0].get_detector_coordinates()
    detector_coordinates = np.array(list(detector_coordinates.values()))
    detector_coordinates[:, : 2] = detector_coordinates[:, : 2] / 2
    detector_coordinates = detector_coordinates.astype(np.uint8)

    samplers = []
    for c in circuits:
        samplers.append(c.compile_detector_sampler())
    del circuits

    for sampler in samplers:
        stim_data_one_p, observable_flips_one_p = [], []
        start = time.time()
        while len(stim_data_one_p) < (batch_size * buffer_size):
            stim_data, observable_flips = sampler.sample(shots=factor * batch_size * buffer_size,
                                                         separate_observables=True)
            # remove empty syndromes:
            non_empty_indices = (np.sum(stim_data, axis=1) != 0)
            stim_data_one_p.extend(stim_data[non_empty_indices, :])
            observable_flips_one_p.extend(observable_flips[non_empty_indices])
        # if there are more non-empty syndromes than necessary
        stim_data_list.append(stim_data_one_p[: batch_size * buffer_size])
        observable_flips_list.append(observable_flips_one_p[: batch_size * buffer_size])
        end = time.time()
        print(f'*** Sampling with factor={factor} ended in {end - start}s', flush=True)
        factor -= 10

    del stim_data_one_p, observable_flips_one_p, stim_data, observable_flips
    print('****************************************', flush=True)
    print(f'len(stim_data_list)={len(stim_data_list)}')
    print(f'len(stim_data_list[0]={len(stim_data_list[0])}', end=', ', flush=True)
    print(f'len(stim_data_list[0][0])={len(stim_data_list[0][0])}')
    print('****************************************\n', flush=True)

    stim_data_list = [val for tup in zip(*stim_data_list) for val in tup]
    observable_flips_list = [val for tup in zip(*observable_flips_list) for val in tup]

    print('****************************************', flush=True)
    print(f'len(stim_data_list)={len(stim_data_list)}', flush=True)
    print(f'len(stim_data_list[0]={len(stim_data_list[0])}', flush=True)
    print('****************************************\n', flush=True)

    repeated_arguments = []
    N_b = batch_size * len(error_rate)
    for i in range(buffer_size):
        repeated_arguments.append((stim_data_list[i * N_b: (i + 1) * N_b],
                                   observable_flips_list[i * N_b: (i + 1) * N_b],
                                   detector_coordinates,
                                   mask, m_nearest_nodes, power))

    del stim_data_list, observable_flips_list

    print('****************************************', flush=True)
    print(f'len(repeated_arguments)={len(repeated_arguments)}', flush=True)
    print('****************************************\n', flush=True)

    with Pool(processes=(cpu_count() - 1)) as pool:
        buffer = pool.starmap(generate_batch, repeated_arguments)

    del repeated_arguments

    print('****************************************', flush=True)
    print(f'len(buffer)={len(buffer)}', flush=True)
    print(f'len(buffer[0])={len(buffer[0])}', flush=True)
    print('****************************************\n', flush=True)

    buffer = [item for sublist in buffer for item in sublist]

    print('****************************************', flush=True)
    print(f'len(buffer)={len(buffer)}')
    print('****************************************', flush=True)
