import gc
import time
import json
import numpy as np
import pandas as pd
import stim
import torch
from torch_geometric.data import Data
from multiprocessing import Pool, cpu_count


def get_node_list_3D(syndrome_3D):
    """
    Create two arrays, one containing the syndrome defects,
    and the other containing their corresponding contiguous
    indices in the matrix representation of the syndrome.
    """
    defect_indices_triple = np.nonzero(syndrome_3D)
    defects = syndrome_3D[defect_indices_triple]
    return defects, defect_indices_triple


def get_node_feature_matrix(defects, defect_indices_triple, num_node_features=None):
    """
    Creates a node feature matrix of dimensions
    (number_of_defects, number_of_node_features), where each row
    is the feature vector of a single node.
    The feature vector is defined as
    x = (X, Z, d_north, d_west, d_time)
        X: 1(0) if defect corresponds to a X(Z) stabilizer
        Z: 1(0) if defect corresponds to a Z(X) stabilizer
        d_north: distance to north boundary, i.e. row index in syndrome matrix
        d_west: distance to west boundary, i.e. column index in syndrome matrix
        d_time: distance in time from the first measurement
    """

    if num_node_features is None:
        num_node_features = 5  # By default, use 4 node features

    # Get defects (non_zero entries), defect indices (indices of defects in
    # flattened syndrome)
    # and defect_indices_tuple (indices in 3D syndrome) of the syndrome matrix

    num_defects = defects.shape[0]

    defect_indices_triple = np.transpose(np.array(defect_indices_triple))

    # get indices of x and z type defects, resp.
    x_defects = (defects == 1)
    z_defects = (defects == 3)

    # initialize node feature matrix
    node_features = np.zeros([num_defects, num_node_features])
    # defect is x type:
    node_features[x_defects, 0] = 1
    # distance of x tpe defect from northern and western boundary:
    node_features[x_defects, 2:] = defect_indices_triple[x_defects, :]

    # defect is z type:
    node_features[z_defects, 1] = 1
    # distance of z tpe defect from northern and western boundary:
    node_features[z_defects, 2:] = defect_indices_triple[z_defects, :]

    return node_features


def get_3D_graph(syndrome_3D,
    target = None,
    m_nearest_nodes = None,
    power = None):
    """
    Form a graph from a repeated syndrome measurement where a node is added,
    each time the syndrome changes. The node features are 5D.
    """
    # get defect indices:
    defects, defect_indices_triple = get_node_list_3D(syndrome_3D)

    # Use helper function to create node feature matrix as torch.tensor
    # (X, Z, N-dist, W-dist, time-dist)
    X = get_node_feature_matrix(defects, defect_indices_triple,
        num_node_features = 5)
    # set default power of inverted distances to 1
    if power is None:
        power = 1.

    # construct the adjacency matrix!
    n_defects = len(defects)
    y_coord = defect_indices_triple[0].reshape(n_defects, 1)
    x_coord = defect_indices_triple[1].reshape(n_defects, 1)
    t_coord = defect_indices_triple[2].reshape(n_defects, 1)

    y_dist = np.abs(y_coord.T - y_coord)
    x_dist = np.abs(x_coord.T - x_coord)
    t_dist = np.abs(t_coord.T - t_coord)

    # inverse square of the supremum norm between two nodes
    Adj = np.maximum.reduce([y_dist, x_dist, t_dist])
    # set diagonal elements to nonzero to circumvent division by zero
    np.fill_diagonal(Adj, 1)
    # scale the edge weights
    Adj = 1./Adj ** power
    # set diagonal elements to zero to exclude self loops
    np.fill_diagonal(Adj, 0)

    # remove all but the m_nearest neighbours
    if m_nearest_nodes is not None:
        for ix, row in enumerate(Adj.T):
            # Do not remove edges if a node has (degree <= m)
            if np.count_nonzero(row) <= m_nearest_nodes:
                continue
            # Get indices of all nodes that are not the m nearest
            # Remove these edges by setting elements to 0 in adjacency matrix
            Adj.T[ix, np.argpartition(row,-m_nearest_nodes)[:-m_nearest_nodes]] = 0.

    Adj = np.maximum(Adj, Adj.T) # Make sure for each edge i->j there is edge j->i
    n_edges = np.count_nonzero(Adj) # Get number of edges

    # get the edge indices:
    edge_index = np.nonzero(Adj)
    edge_attr = Adj[edge_index].reshape(n_edges, 1)
    edge_index = np.array(edge_index)

    if target is not None:
        y = target.reshape(1, 1)
    else:
        y = None

    return [X.astype(np.float32), edge_index.astype(np.int64,), edge_attr.astype(np.float32), y.astype(np.float32)]


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


def generate_batch(stim_data_list,
                   observable_flips_list,
                   detector_coordinates,
                   mask, m_nearest_nodes, power):
    '''
    Generates a batch of graphs from a list of stim experiments.
    '''
    batch = []

    # start_time = time.time()

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
        graph = {'x': graph[0].tolist(), 'edge_index': graph[1].tolist(),
                 'edge_attr': graph[2].tolist(), 'y': graph[3].tolist()}
        batch.append(graph)
    return batch


def syndrome_mask(code_size, repetitions):
    '''
    Creates a surface code grid. 1: X-stabilizer. 3: Z-stabilizer.
    '''
    M = code_size + 1

    syndrome_matrix_X = np.zeros((M, M), dtype=np.uint8)

    # starting from northern boundary:
    syndrome_matrix_X[::2, 1:M - 1:2] = 1

    # starting from first row inside the grid:
    syndrome_matrix_X[1::2, 2::2] = 1

    syndrome_matrix_Z = np.rot90(syndrome_matrix_X) * 3
    # Combine syndrome matrices where 1 entries
    # correspond to x and 3 entries to z defects
    syndrome_matrix = (syndrome_matrix_X + syndrome_matrix_Z)

    # Return the syndrome matrix
    return np.dstack([syndrome_matrix] * (repetitions + 1))


def generate_buffer(buffer_size, code_size, repetitions, error_rate, perfect_stabilizer: bool):
    '''
    Creates a buffer with len(error_rate)*batch_size*buffer_size samples.
    Empty syndromes are removed from the training data, because empty graphs
    can't be handled by PyTorch and should be easily classified as I.
    '''
    print(f"=== code_size={code_size} repetitions={repetitions} ===", flush=True)

    m_nearest_nodes = 6
    power = 2
    batch_size = 1000
    test_size = 10
    circuits = []
    for p in error_rate:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=repetitions,
            distance=code_size,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p)
        circuits.append(circuit)
    detector_coordinates = circuits[0].get_detector_coordinates()
    detector_coordinates = np.array(list(detector_coordinates.values()))
    detector_coordinates[:, : 2] = detector_coordinates[:, : 2] / 2
    detector_coordinates = detector_coordinates.astype(np.uint8)
    samplers = []
    for circuit in circuits:
        sampler = circuit.compile_detector_sampler()
        samplers.append(sampler)
    mask = syndrome_mask(code_size, repetitions)

    stim_data_list, observable_flips_list = [], []

    # repeat each experiments multiple times to get enough non-empty
    # syndromes. This number decreases with increasing p
    start = time.time()
    factor = 50 if not perfect_stabilizer else 40
    # sample for each error rate:
    for sampler in samplers:
        stim_data_one_p, observable_flips_one_p = [], []
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
        # decrease the number of samples with increasing p:
        factor -= 10
    del stim_data_one_p, observable_flips_one_p, stim_data, observable_flips

    # interleave lists to mix error rates:
    # [sample(p1), sample(p2), ..., sample(p_n), sample(p1), sample(p2), ...]
    stim_data_list = [val for tup in zip(*stim_data_list) for val in tup]
    observable_flips_list = [val for tup in zip(*observable_flips_list) for val in tup]

    # len of single batches:
    # N_b = no_samples / buffer_size = len(error_rate) * batch_size
    repeated_arguments = []
    N_b = batch_size * len(error_rate)
    for i in range(buffer_size):
        repeated_arguments.append((stim_data_list[i * N_b: (i + 1) * N_b],
                                   observable_flips_list[i * N_b: (i + 1) * N_b],
                                   detector_coordinates,
                                   mask, m_nearest_nodes, power))

    del stim_data_list, observable_flips_list
    gc.collect()

    print(f'Generation ended in {time.time() - start}s.', flush=True)

    conv_start = time.time()
    # create batches in parallel:
    with Pool(processes=24) as pool:
        buffer = pool.starmap(generate_batch, repeated_arguments)
    del repeated_arguments
    # flatten the buffer:
    buffer = pd.DataFrame([item for sublist in buffer for item in sublist])
    val_buffer = buffer.iloc[:(test_size * batch_size * len(error_rate)), :]
    buffer = buffer.iloc[(test_size * batch_size * len(error_rate)):, :]
    print(f'Conversion to Data object ended in {time.time() - conv_start}s.', flush=True)
    gc.collect()

    #buffer = []
    #for elem in repeated_arguments:
    #    buffer.append(generate_batch(elem[0], elem[1], elem[2], elem[3], elem[4], elem[5]))
    #buffer = [item for sublist in buffer for item in sublist]

    # convert list of numpy arrays to torch Data object containing torch GPU tensors
    #torch_buffer = []
    #for i in range(len(buffer)):
        #X = torch.from_numpy(buffer[i][0])
        #edge_index = torch.from_numpy(buffer[i][1])
        #edge_attr = torch.from_numpy(buffer[i][2])
        #y = torch.from_numpy(buffer[i][3])
        #torch_buffer.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y))
    #del buffer
    save_start = time.time()
    if not perfect_stabilizer:
        #torch.save(buffer[(test_size * batch_size * len(error_rate)):],
        #           f"circuit_level_noise/d{code_size}/d{code_size}_dt{repetitions}_training_set.pt")
        #torch.save(buffer[:(test_size * batch_size * len(error_rate))],
        #           f"circuit_level_noise/d{code_size}/d{code_size}_dt{repetitions}_validation_set.pt")
        #with open(f"circuit_level_noise/d{code_size}/d{code_size}_dt{repetitions}_training_set.json", 'w') as f:
        #    json.dump(buffer[(test_size * batch_size * len(error_rate)):], f)
        #with open(f"circuit_level_noise/d{code_size}/d{code_size}_dt{repetitions}_validation_set.json", 'w') as f:
        #    json.dump(buffer[:(test_size * batch_size * len(error_rate))], f)
        buffer.to_json(f"circuit_level_noise/d{code_size}/d{code_size}_dt{repetitions}_training_set.json",
                       orient='records', lines=True)
        val_buffer.to_json(f"circuit_level_noise/d{code_size}/d{code_size}_dt{repetitions}_validation_set.json",
                           orient='records', lines=True)

    else:
        #torch.save(buffer[(test_size * batch_size * len(error_rate)):],
        #           f"perfect_stabilizers/d{code_size}/d{code_size}_dt{repetitions}_ps_training_set.pt")
        #torch.save(buffer[:(test_size * batch_size * len(error_rate))],
        #           f"perfect_stabilizers/d{code_size}/d{code_size}_dt{repetitions}_ps_validation_set.pt")
        #with open(f"perfect_stabilizers/d{code_size}/d{code_size}_dt{repetitions}_ps_training_set.json", 'w') as f:
        #    json.dump(buffer[(test_size * batch_size * len(error_rate)):], f)
        #with open(f"perfect_stabilizers/d{code_size}/d{code_size}_dt{repetitions}_ps_validation_set.json", 'w') as f:
        #    json.dump(buffer[:(test_size * batch_size * len(error_rate))], f)
        buffer.to_json(f"perfect_stabilizers/d{code_size}/d{code_size}_dt{repetitions}_ps_training_set.json",
                       orient='records', lines=True)
        val_buffer.to_json(f"perfect_stabilizers/d{code_size}/d{code_size}_dt{repetitions}_ps_validation_set.json",
                           orient='records', lines=True)
    print(f'Saved in {time.time() - save_start}s.', flush=True)
    del buffer, val_buffer


buffer_size = 1000
test_size = 10

'''
print('*** CHUNK SIZE = 10^6 ***\ninizio lettura', flush=True)
start = time.time()
df = pd.read_json('circuit_level_noise/d5/d5_dt9_training_set.json', orient='records', lines=True, chunksize=10**3)
print(f'fine lettura. Tempo {time.time() - start}s\ninizio composizione', flush=True)
start = time.time()
df = pd.concat(df, ignore_index=True)
print(f'fine composizione. Tempo {time.time() - start}s', flush=True)
del df

print('*** CHUNK SIZE = 10^4 ***\ninizio lettura', flush=True)
start = time.time()
df = pd.read_json('circuit_level_noise/d5/d5_dt9_training_set.json', orient='records', lines=True, chunksize=100)
print(f'fine lettura. Tempo {time.time() - start}s\ninizio composizione', flush=True)
start = time.time()
df = pd.concat(df, ignore_index=True)
print(f'fine composizione. Tempo {time.time() - start}s', flush=True)
exit(0)'''

# Circuit level noise:
# sizes, reps, error_rate = [5, 7], [3, 5, 7, 9, 11], [0.001, 0.002, 0.003, 0.004, 0.005]
# sizes, reps, error_rate = [5], [11], [0.001, 0.002, 0.003, 0.004, 0.005]

# Perfect stabilizers:
# sizes, reps, error_rate = [5, 7, 9, 11, 13, 15], [1], [0.01, 0.05, 0.10, 0.15]
sizes, reps, error_rate = [9], [1], [0.01, 0.05, 0.10, 0.15]

# !! sizes 5 with dt = [3, 5, 7] are 'split' not 'records' !!
for code_size in sizes:
    for repetitions in reps:
        generate_buffer(buffer_size=buffer_size + test_size, code_size=code_size, repetitions=repetitions,
                        error_rate=error_rate, perfect_stabilizer=True if len(error_rate) == 4 else False)
        gc.collect()

