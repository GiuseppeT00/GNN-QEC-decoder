import os
import gc
import pymatching
import stim
import torch
import numpy as np
import pandas as pd
import src.gnn_models as gnn
from time import time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from multiprocessing import Pool


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
                 target=None,
                 m_nearest_nodes=None,
                 power=None):
    """
    Form a graph from a repeated syndrome measurement where a node is added,
    each time the syndrome changes. The node features are 5D.
    """
    # get defect indices:
    defects, defect_indices_triple = get_node_list_3D(syndrome_3D)

    # Use helper function to create node feature matrix as torch.tensor
    # (X, Z, N-dist, W-dist, time-dist)
    X = get_node_feature_matrix(defects, defect_indices_triple, num_node_features=5)
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
    Adj = 1. / Adj ** power
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
            Adj.T[ix, np.argpartition(row, -m_nearest_nodes)[:-m_nearest_nodes]] = 0.

    Adj = np.maximum(Adj, Adj.T)  # Make sure for each edge i->j there is edge j->i
    n_edges = np.count_nonzero(Adj)  # Get number of edges

    # get the edge indices:
    edge_index = np.nonzero(Adj)
    edge_attr = Adj[edge_index].reshape(n_edges, 1)
    edge_index = np.array(edge_index)

    if target is not None:
        y = target.reshape(1, 1)
    else:
        y = None

    return [X.astype(np.float32), edge_index.astype(np.int64, ), edge_attr.astype(np.float32), y.astype(np.float32)]


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


def generate_buffer(sampler: stim.CompiledDetectorSampler, test_size: int) -> (list, list, int):
    print("===== Starting data generation =====", flush=True)
    correct_predictions_trivial = 0
    stim_data_list, observable_flips_list = [], []
    stim_data, observable_flips = sampler.sample(shots=test_size, separate_observables=True)
    non_empty_indices = (np.sum(stim_data, axis=1) != 0)
    stim_data_list.extend(stim_data[non_empty_indices, :])
    observable_flips_list.extend(observable_flips[non_empty_indices])
    correct_predictions_trivial += len(observable_flips[~ non_empty_indices])
    stim_data_list = stim_data_list[: test_size]
    observable_flips_list = observable_flips_list[: test_size]
    print(f"===== Correctly generated {len(stim_data_list) + correct_predictions_trivial} test samples =====",
          flush=True)
    gc.collect()
    return stim_data_list, observable_flips_list, correct_predictions_trivial


def decode_mwpm_data(circuit: stim.Circuit, syndrome: np.ndarray, actual_observables: np.ndarray,
                     correct_predictions_trivial: int) -> (float, float):
    print("===== Starting decoding data with MWPM =====", flush=True)
    matching = circuit.detector_error_model(decompose_errors=True)
    elapsed_time = 0
    del circuit
    matching = pymatching.Matching.from_detector_error_model(matching)
    correct_predictions = 0
    for i in range(syndrome.shape[0]):
        start = time()
        predicted_observables = matching.decode(syndrome[i, :])
        elapsed_time += (time() - start)
        correct_predictions += 1 if np.array_equal(actual_observables[i, :], predicted_observables) else 0
    accuracy = (correct_predictions + correct_predictions_trivial) / (syndrome.shape[0] + correct_predictions_trivial)
    print(f"===== Finished decoding data with MWPM (Accuracy {accuracy}, Elapsed time {elapsed_time}) =====",
          flush=True)
    return elapsed_time, accuracy


def decode_gnn_data(stim_data_list: list, observable_flips_list: list, detector_coordinates: np.ndarray,
                    mask: np.ndarray, correct_predictions_trivial: int, code_size: int, repetitions: int) -> (float, float):
    print("===== Starting decoding data with GNN decoder =====", flush=True)
    power = 2
    m_nearest_nodes = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = gnn.GNN_7(num_classes=1, manual_seed=12345, num_node_features=5)
    # loaded_weights = torch.load(f'results/circuit_level_noise/d{code_size}/d{code_size}_d_t_{repetitions}'
    #                             f'_rep_epoch{100 if code_size == 3 else 600}.pt', map_location=device)['model']
    loaded_weights = torch.load(f'results/perfect_stabilizers/d{code_size}/d{code_size}_d_t_{repetitions}'
                                f'_rep_epoch600.pt', map_location=device)['model']
    model.load_state_dict(loaded_weights)
    model = model.to(device)
    sigmoid = torch.nn.Sigmoid()
    print("== Generating buffer and creating DataLoader object ==", flush=True)
    buffer = generate_batch(stim_data_list, observable_flips_list,
                            detector_coordinates, mask, m_nearest_nodes, power)
    del stim_data_list, observable_flips_list
    buffer_samples = len(buffer)
    buffer = [Data(x=torch.tensor(graph['x'], device=device),
                   edge_index=torch.tensor(graph['edge_index'], device=device, dtype=torch.int64),
                   edge_attr=torch.tensor(graph['edge_attr'], device=device),
                   y=torch.tensor(graph['y'], device=device)) for graph in buffer]
    loader = DataLoader(buffer, batch_size=1000)
    print("== Task completed, starting testing ==", flush=True)
    del buffer
    correct_predictions = 0
    elapsed_time = 0
    model.eval()  # run network in training mode
    with torch.no_grad():  # turn off gradient computation (https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
        for data in loader:
            start = time()
            # Perform forward pass to get network output
            data.batch = data.batch.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            target = data.y.to(int)  # Assumes binary targets (no probabilities)
            # Sum correct predictions
            prediction = sigmoid(out.detach()).round().to(int)
            elapsed_time += (time() - start)
            correct_predictions += int((prediction == target).sum())
    accuracy = (correct_predictions + correct_predictions_trivial) / (buffer_samples + correct_predictions_trivial)
    print(f"===== Finished decoding data with GNN (Accuracy {accuracy}, Elapsed time {elapsed_time}) =====",
          flush=True)
    return elapsed_time, accuracy


def testing(code_size: int, repetitions: int, error_rate: float, test_size: int):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=repetitions,
        distance=code_size,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate,
        before_round_data_depolarization=error_rate)
    detector_coordinates = circuit.get_detector_coordinates()
    detector_coordinates = np.array(list(detector_coordinates.values()))
    detector_coordinates[:, : 2] = detector_coordinates[:, : 2] / 2
    detector_coordinates = detector_coordinates.astype(np.uint8)
    sampler = circuit.compile_detector_sampler()
    mask = syndrome_mask(code_size, repetitions)

    stim_data_list, observable_flips_list, correct_predictions_trivial = generate_buffer(sampler, test_size)
    print(f"== Trivial predictions {correct_predictions_trivial} ==", flush=True)
    mwpm_elapsed_time, mwpm_accuracy = decode_mwpm_data(circuit, np.array(stim_data_list),
                                                        np.array(observable_flips_list), correct_predictions_trivial)
    gc.collect()
    gnn_elapsed_time, gnn_accuracy = decode_gnn_data(stim_data_list, observable_flips_list, detector_coordinates,
                                                     mask, correct_predictions_trivial, code_size, repetitions)
    del stim_data_list, observable_flips_list
    gc.collect()
    print(f"===== Benchmark =====", flush=True)
    print(f"=== MWPM: Accuracy {mwpm_accuracy} , Logical failure rate {1 - mwpm_accuracy} , Average elapsed time "
          f"{mwpm_elapsed_time / test_size}) , Total elapsed time {mwpm_elapsed_time} ===", flush=True)
    print(f"=== GNN: Accuracy {gnn_accuracy}, Logical failure rate {1 - gnn_accuracy} , Average elapsed time "
          f"{gnn_elapsed_time / test_size}) , Total elapsed time {gnn_elapsed_time} ===", flush=True)


if __name__ == '__main__':
    # test_size = 5 * (10 ** 6)
    test_size = 10 ** 6
    # test_sizes = [10 ** 3, 10 ** 3, 10 ** 3, 10 ** 3, 10 ** 3]
    # error_rate = 0.003
    error_rate = [0.01, 0.05, 0.10, 0.15, 0.20]
    # code_sizes = [3, 5, 7]
    code_sizes = [5, 7, 9]
    # repetitions = [3, 5, 7, 9, 11]
    repetitions = [1]
    # IN CASO DI PERFECT STABILIZERS CAMBIARE PATH IN load_state_dict

    if isinstance(error_rate, list):
        for error in error_rate:
            for code_size in code_sizes:
                for i in range(len(repetitions)):
                    print(f'========== STARTING TESTING CODE SIZE {code_size} , REPETITIONS {repetitions[i]} ,'
                          f' ERROR RATE {error_rate} , TESTING SAMPLES {test_size} ==========', flush=True)
                    testing(code_size, repetitions[i], error, test_size)
                    gc.collect()
                    print("\n\n", flush=True)
    else:
        for code_size in code_sizes:
            for i in range(len(repetitions)):
                print(f'========== STARTING TESTING CODE SIZE {code_size} , REPETITIONS {repetitions[i]} ,'
                      f' ERROR RATE {error_rate} , TESTING SAMPLES {test_size} ==========', flush=True)
                testing(code_size, repetitions[i], error_rate, test_size)
                gc.collect()
                print("\n\n", flush=True)
