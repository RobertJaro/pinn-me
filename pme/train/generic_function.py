import os.path
from multiprocessing import Pool

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.special import voigt_profile, wofz
from torch import nn

from pme.model import Sine


class GenericFunction(nn.Module):

    def __init__(self, input_parameters, polynomial_degree=10):
        super().__init__()
        # self.weights = nn.Parameter(torch.randn(input_parameters, polynomial_degree), requires_grad=True)
        self.polynomial_degree = polynomial_degree
        self.input_layer = nn.Linear(input_parameters, 64)
        self.hidden_layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(6)])
        self.output_layer = nn.Linear(64, 1)
        self.activation = Sine()

    def forward(self, params):
        # polynomial = torch.stack([params ** i for i in range(self.polynomial_degree)], dim=-1)
        # profile = torch.einsum('ijk,jk->i', polynomial, self.weights)
        x = self.input_layer(params)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        profile = self.output_layer(x)
        return profile[..., 0]


def fit(function, base_path, name, batch_size=1024 * 8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    approx_function = GenericFunction(3)
    approx_function.to(device)

    nu = np.linspace(-50, 50, 1000, dtype=np.float32)
    gamma = np.linspace(0, 50, 100, dtype=np.float32)
    mu = np.linspace(-50, 50, 100, dtype=np.float32)
    nu, gamma, mu = np.meshgrid(nu, gamma, mu, indexing='ij')
    nu = nu.flatten()
    gamma = gamma.flatten()
    mu = mu.flatten()
    sigma = np.ones_like(nu, dtype=np.float32)

    with Pool(16) as p:
        profile = p.starmap(function, zip(nu, sigma, gamma, mu))
    profile = np.array(profile, dtype=np.float32)

    profile = torch.tensor(profile, dtype=torch.float32)
    params = torch.tensor(np.stack([nu, gamma, mu], axis=-1), dtype=torch.float32)

    # shuffle
    idx = torch.randperm(len(profile))
    profile = profile[idx]
    params = params[idx]

    # optimizer
    optimizer = torch.optim.Adam(approx_function.parameters(), lr=1e-3)

    n_batches = np.ceil(len(profile) / batch_size).astype(int)

    for epoch in range(100):
        total_loss = []
        for i in range(n_batches):
            batch = profile[i * batch_size:(i + 1) * batch_size]
            batch_params = params[i * batch_size:(i + 1) * batch_size]
            batch_params.requires_grad = True

            batch, batch_params = batch.to(device), batch_params.to(device)

            batch_profile = approx_function.forward(batch_params)

            loss = torch.mean((batch_profile - batch) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += [loss.cpu().detach().numpy()]
        print(f'Epoch {epoch + 1}, loss {np.mean(total_loss)}')

    torch.save(approx_function, os.path.join(base_path, f'{name}.pt'))

    # test plot
    for t_gamma, t_mu in [(0, 0), (1, 10), (1, -10), (10, 0), (10, 10), (10, -10), (40, 0), (40, 10), (40, -10)]:
        test_nu = np.linspace(-50, 50, 100)
        test_gamma = np.ones_like(test_nu) * t_gamma
        test_mu = np.ones_like(test_nu) * t_mu
        test_sigma = np.ones_like(test_nu)

        with Pool(16) as p:
            test_profile = p.starmap(function, zip(test_nu, test_sigma, test_gamma, test_mu))
        test_profile = np.array(test_profile, dtype=np.float32)
        test_param = torch.tensor(np.stack([test_nu, test_gamma, test_mu], axis=-1), dtype=torch.float32)
        pred_profile = approx_function.forward(test_param.to(device)).detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(test_nu, test_profile, label='ref')
        ax.plot(test_nu, pred_profile, label='pred', linestyle='--')
        ax.legend()
        plt.savefig(os.path.join(base_path, f'{name}_{t_gamma:02d}_{t_mu:02d}.jpg'), dpi=150)
        plt.close()


def faraday_voigt(nu, sigma, gamma, mu):
    ''' Compute the Faraday-Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''

    def z(x, sigma, gamma, mu):
        gamma_i = complex(0, gamma)
        return (x - mu + gamma_i) / (2 ** 0.5 * sigma)

    z_arr = z(nu, sigma, gamma, mu)
    z11 = wofz(z_arr)
    psi_profile = -1 * z11.imag / 1.772

    return psi_profile


def voigt(nu, sigma, gamma, mu):
    ''' Compute the Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''
    phi_profile = voigt_profile(nu - mu, sigma, gamma) * 1.414
    return phi_profile


if __name__ == '__main__':
    base_path = '/glade/work/rjarolim/pinn_me/profile/'
    # fit(voigt, base_path, 'voigt')
    fit(faraday_voigt, base_path, 'faraday_voigt')
