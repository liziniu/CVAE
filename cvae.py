import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CVAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, label_dims=0):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(encoder_layer_sizes, latent_size, label_dims)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, label_dims)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, c=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(device)

        recon_x = self.decoder(z, c)

        return recon_x

    def loss_fn(self, recon_x, x, mean, log_var):

        mse = F.mse_loss(recon_x, x)
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (mse + kld) / x.size(0)

    def update(self, dataset, batch_size, epochs):

        start_time = time.time()
        x_data = torch.tensor(torch.from_numpy(dataset.action), dtype=torch.float32)
        y_data = torch.tensor(torch.from_numpy(dataset.state), dtype=torch.float32)
        train_data = TensorDataset(x_data, y_data)
        data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        loss = 0.
        loss_info = []
        for epoch in range(epochs):

            for i, (x, y) in enumerate(data_loader):
                x, y = x.to(device), y.to(device)
                recon_x, mean, log_var, z = self(x, y)

                loss = self.loss_fn(recon_x, x, mean, log_var)
                loss_info.append(loss.cpu().data.numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("CVAE Epoch Loss: ", loss.cpu().data.numpy())

        print("CVAE Train Time: ", time.time() - start_time)
        return loss_info


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, label_dims):
        super().__init__()

        layer_sizes[0] += label_dims

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.means = nn.Linear(layer_sizes[-1], latent_size)
        self.log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        x = self.MLP(torch.cat((x, c), dim=-1))

        means = self.means(x)
        log_vars = self.log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, label_dims):
        super().__init__()

        input_size = latent_size + label_dims

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="Tanh", module=nn.Tanh())

    def forward(self, z, c):

        x = self.MLP(torch.cat((z, c), dim=-1))

        return x

