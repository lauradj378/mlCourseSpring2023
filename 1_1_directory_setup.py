"""
In this homework, you setup a Python project folder that will be useful for future assignments.

1. Download the file `dl.zip` from Webcourses -> Files -> Python
2. Read the included `readme.md`. It explains the reasoning and setup of your Python environment to run files.
3. Run the included example `assignments/1_1_directory_setup.py`. It is the introductory problem from class, split into several files.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import dl
import dl.data
import dl.networks
import dl.models


torch.manual_seed(0)
params = {
    'width': 50,
    'lr': 0.1,
    'epochs': 25,
    'n_samples': 100,
    'batch_size': 10,
}


if __name__ == '__main__':

    dataset = dl.data.parabola_1d(params['n_samples'])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params['batch_size'], shuffle=True)

    in_dim = dataset.data.shape[-1]
    out_dim = dataset.targets.shape[-1]
    network = dl.networks.Shallow(in_dim, out_dim, width=params['width'])
    optimizer = torch.optim.SGD(network.parameters(), lr=params['lr'])
    model = dl.models.Model(network, optimizer, F.mse_loss, params['epochs'])
    _, loss = model.fit(dataloader)

    x = dataset.data
    y = dataset.targets
    pred = model.predict(x)
    plt.plot(x.detach().numpy().flat, y.detach().numpy(), color='red')
    plt.plot(x.detach().numpy().flat, pred.detach().numpy(), color='blue')
    plt.show()
