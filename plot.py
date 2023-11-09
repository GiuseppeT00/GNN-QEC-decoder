from typing import Union
import torch
import matplotlib.pyplot as plt
import numpy as np


def training_plot(data_path: list[str]):
    models = list()
    colors = list()
    if len(data_path) == 2:
        models = ['Fixed dataset', 'Replacing with new data']
        colors = ['darkorange', 'turquoise']

    elif len(data_path) == 5:
        models = ['d_t = 3', 'd_t = 5', 'd_t = 7', 'd_t = 9', 'd_t = 11']
        colors = ['turquoise', 'darkorange', 'royalblue', 'orchid', 'yellowgreen']
    data = [torch.load(path, map_location=torch.device('cpu'))['training_history'] for path in data_path]
    epochs = [epoch for epoch in range(len(data[0]['loss']))]
    plt.plot([], [], color='black', label='Training')
    plt.plot([], [], ':', color='black', label='Test')
    for i in range(len(models)):
        plt.plot(epochs, data[i]['accuracy'], color=colors[i], label=models[i])
        plt.plot(epochs, data[i]['val_acc'], ':', color=colors[i])
    if len(models) == 5:
        plt.xlim([0, len(epochs) + 1])
        plt.xticks(np.arange(0, len(epochs) + 1, step=100))
        plt.ylim([0.95, 1])
        # plt.yticks(np.arange(0.5, 1, step=0.1))
    elif len(models) == 2:
        plt.xlim([0, len(epochs) + 1])
        plt.xticks(np.arange(0, len(epochs) + 1, step=50))
        plt.ylim([0.995, 1])
        plt.yticks(np.arange(0.98, 1, step=0.005))
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot([], [], color='black', label='Training')
    plt.plot([], [], ':', color='black', label='Test')
    for i in range(len(models)):
        plt.plot(epochs, data[i]['loss'], color=colors[i], label=models[i])
        plt.plot(epochs, data[i]['val_loss'], ':', color=colors[i])
    plt.xlim([0, len(epochs) + 1])
    plt.ylim([0, 0.1])
    plt.legend()
    plt.show()


def plot_over_error_rate_plot_ps(accuracy: bool):
    data = [
        {
            'MWPM': [0.0143, 0.3433, 0.4877, 0.49946, 0.4993],
            'GNN': [0.01426, 0.3287, 0.48328, 0.4995, 0.498796]
        },
        {
            'MWPM': [0.0081, 0.3849, 0.4966, 0.49935, 0.499566],
            'GNN': [0.0084, 0.3763, 0.4965, 0.49988, 0.49948]
        },
        {
            'MWPM': [0.00415, 0.4175, 0.4995, 0.500136, 0.50019],
            'GNN': [0.004977, 0.4166, 0.499, 0.500387, 0.49882]
        }
    ]
    error_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    models = ['d = 5', 'd = 7', 'd = 9']
    colors = ['turquoise', 'darkorange', 'royalblue', 'orchid', 'yellowgreen', 'yellow']
    plt.plot([], [], color='black', label='GNN')
    plt.plot([], [], ':', color='black', label='MWPM')
    for i in range(len(data)):
        plt.plot(error_rates, data[i]['MWPM'] if not accuracy else [1 - lfr for lfr in data[i]['MWPM']], ':', color=colors[i], label=models[i])
        plt.plot(error_rates, data[i]['GNN'] if not accuracy else [1 - lfr for lfr in data[i]['GNN']], color=colors[i], label=models[i])
    plt.xlim([0.0, 0.2])
    plt.xticks(error_rates)
    plt.ylim([0.0, 0.5] if not accuracy else [0.5, 1])
    #plt.yticks(np.arange(0.0, 0.2, step=0.05))
    plt.legend()
    plt.show()


def plot_over_error_rate_plot_cln(accuracy: bool):
    data = [
        {
            'MWPM': [0.0065006, 0.0105776, 0.014484, 0.0182384, 0.0222814],
            'GNN': [0.0058776, 0.0088658, 0.011714, 0.0146828, 0.0178594]
        },
        {
            'MWPM': [0.00188, 0.003372, 0.0047338, 0.00612, 0.0076556],
            'GNN': [0.00143, 0.00244, 0.0034626, 0.005149, 0.006574]
        }
    ]
    models = ['dt = 3', 'dt = 5', 'dt = 7', 'dt = 9', 'dt = 11']
    distances = [3, 5, 7, 9, 11]
    colors = ['turquoise', 'darkorange', 'royalblue', 'orchid', 'yellowgreen', 'yellow']
    plt.plot([], [], color='black', label='GNN')
    plt.plot([], [], ':', color='black', label='MWPM')
    for i in range(len(data)):
        plt.plot(distances, data[i]['MWPM'] if not accuracy else [1 - lfr for lfr in data[i]['MWPM']], ':', color=colors[i], label=models[i])
        plt.plot(distances, data[i]['GNN'] if not accuracy else [1 - lfr for lfr in data[i]['GNN']], color=colors[i], label=models[i])
    # plt.xlim([0.0, 0.2])
    plt.xticks(distances)
    plt.ylim([0.0, 0.025] if not accuracy else [0.975, 1])
    #plt.yticks(np.arange(0.0, 0.2, step=0.05))
    plt.legend()
    plt.show()


'''
training_plot(['results/circuit_level_noise/d5/d5_d_t_3_rep_epoch600.pt',
               'results/circuit_level_noise/d5/d5_d_t_5_rep_epoch600.pt',
               'results/circuit_level_noise/d5/d5_d_t_7_rep_epoch600.pt',
               'results/circuit_level_noise/d5/d5_d_t_9_rep_epoch600.pt',
               'results/circuit_level_noise/d5/d5_d_t_11_rep_epoch600.pt'])
'''


plot_over_error_rate_plot_ps(accuracy=False)
plot_over_error_rate_plot_ps(accuracy=True)

'''
plot_over_error_rate_plot_cln(accuracy=False)
plot_over_error_rate_plot_cln(accuracy=True)
'''

'''
data = torch.load('results/perfect_stabilizers/d9_d_t_1_rep_epoch600.pt', map_location=torch.device('cpu'))['training_history']
epochs = [epoch for epoch in range(len(data['loss']))]
plt.plot(epochs, data['accuracy'], color='red', label='Train')
plt.plot(epochs, data['val_acc'], ':', color='blue', label='Val')
plt.ylim([0.615, 0.65])
plt.legend()
plt.show()
plt.clf()
plt.plot(epochs, data['loss'], color='red', label='Train')
plt.plot(epochs, data['val_loss'], ':', color='blue', label='Val')
plt.ylim([0.513, 0.535])
plt.legend()
plt.show()
'''
