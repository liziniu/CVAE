import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kl_divergence(mean1, log_var1, mean2, log_var2):
    """KL Divergence between D_KL[p(x1|mean1, log_var1)||p(x2|mean2, log_var2)]
    Input orders are important
    Reference baselines.common.distribution.DiagGaussianPd
    """
    # return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) \
    #  / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    std1 = torch.exp(log_var1).to(device)
    std2 = torch.exp(log_var2).to(device)
    return torch.sum(log_var2 - log_var1 + (torch.pow(std1, 2) + torch.pow(mean1 - mean2, 2)) / (2.0 * torch.pow(std2, 2)) - 0.5, dim=-1)


def gaussian_log_density(mean, log_var, x):
    """Gaussian Log density calculation"""
    std = torch.exp(log_var).to(device)
    # return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
    #            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
    #            + tf.reduce_sum(self.logstd, axis=-1)
    neg_log_density = 0.5 * torch.sum(torch.pow((x - mean)/(std + 1e-6), 2), dim=-1) + \
                      0.5 * np.log(2.0 * np.pi) * x.size(-1) + \
                      torch.sum(log_var, dim=-1)
    return -neg_log_density


class CVAE(nn.Module):
    def __init__(self, input_size, output_size, latent_size, encoder_layer_sizes, decoder_layer_sizes,
                 alpha=1., dataset_name='mnist'):
        super().__init__()

        self.latent_size = latent_size
        self.alpha = alpha

        self.train_encoder = TrainPriorEncoder(input_size, output_size, latent_size, encoder_layer_sizes)
        self.test_encoder = TestPriorEncoder(input_size, latent_size, encoder_layer_sizes)
        self.decoder = Decoder(input_size, latent_size, output_size, decoder_layer_sizes, dataset_name)
        self.dataset_name = dataset_name

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, y):

        batch_size = x.size(0)

        train_mean, train_log_var = self.train_encoder(x, y)
        test_mean, test_log_var = self.test_encoder(x)

        train_std = torch.exp(train_log_var).to(device)
        train_eps = torch.randn([batch_size, self.latent_size]).to(device)
        train_z = train_eps * train_std + train_mean

        test_std = torch.exp(test_log_var).to(device)
        test_eps = torch.randn([batch_size, self.latent_size]).to(device)
        test_z = test_eps * test_std + test_mean

        train_y = self.decoder(x, train_z)
        test_y = self.decoder(x, test_z)

        return train_mean, train_log_var, train_z, train_y, test_mean, test_log_var, test_z, test_y

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(device)

        recon_x = self.decoder(z, c)

        return recon_x

    def loss_fn(self, recon_x, x, mean, log_var):

        mse = F.mse_loss(recon_x, x)
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (mse + kld) / x.size(0)

    def update(self, x, y):
        x = torch.tensor(torch.from_numpy(x), dtype=torch.float32)
        y = torch.tensor(torch.from_numpy(y), dtype=torch.float32)

        x, y = x.to(device), y.to(device)
        train_mean, train_log_var, train_z, train_y, test_mean, test_log_var, test_z, test_y = self(x, y)
        kl_loss = torch.mean(kl_divergence(train_mean, train_log_var, test_mean, test_log_var), dim=-1)
        if self.dataset_name == 'mnist':
            recon_loss = F.binary_cross_entropy_with_logits(input=train_y, target=y, size_average=True)
            gsnn_loss = F.binary_cross_entropy_with_logits(input=test_y, target=y, size_average=True)
        elif self.dataset_name == 'gmm':
            train_target_mean, train_target_log_var = train_y
            recon_loss = -torch.mean(gaussian_log_density(train_target_mean, train_target_log_var, y), dim=-1)
            test_target_mean, test_target_log_var = test_y
            gsnn_loss = -torch.mean(gaussian_log_density(test_target_mean, test_target_log_var, y), dim=-1)
        else:
            raise NotImplementedError
        loss = (kl_loss + recon_loss) * self.alpha + (1.-self.alpha) * gsnn_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def sample(self, x):
        x = torch.tensor(torch.from_numpy(x), dtype=torch.float32)
        batch_size = x.size(0)
        test_mean, test_log_var = self.test_encoder(x)
        test_std = torch.exp(test_log_var).to(device)
        test_eps = torch.randn([batch_size, self.latent_size]).to(device)
        test_z = test_eps * test_std + test_mean
        test_y = self.decoder(x, test_z)
        if self.dataset_name == 'mnist':
            return torch.sigmoid(test_y)
        elif self.dataset_name == 'gmm':
            return test_y
        else:
            raise NotImplementedError


class TestPriorEncoder(nn.Module):
    """ p(z|x) used when test
    """
    def __init__(self, input_size, latent_size, layer_sizes=(64, 64)):
        super().__init__()

        # (action_dim, 64, 64, latent_size)
        # (action_dim, 64), (64, 64), (64, latent_size)
        layer_sizes = (input_size, ) + tuple(layer_sizes)
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            print('test encoder add fc:{}'.format((in_size, out_size)))

        self.means = nn.Linear(layer_sizes[-1], latent_size)
        self.log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):

        x = self.MLP(x)

        means = self.means(x)
        log_vars = self.log_var(x)

        return means, log_vars


class TrainPriorEncoder(nn.Module):
    """ p(z|x, y) used when train
    """
    def __init__(self, input_size, output_size, latent_size, layer_sizes=(64, 64)):
        super().__init__()

        layer_sizes = (input_size+output_size, ) + tuple(layer_sizes)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            print('train encoder add fc:{}'.format((in_size, out_size)))

        self.means = nn.Linear(layer_sizes[-1], latent_size)
        self.log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, y):

        x = self.MLP(torch.cat((x, y), dim=-1))

        means = self.means(x)
        log_vars = self.log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    """p(y|x, z) when generate
    """

    def __init__(self, input_size, latent_size, output_size, layer_sizes=(64, 64), dataset_name='mnist'):
        super().__init__()
        self._dataset_name = dataset_name
        layer_sizes = (input_size + latent_size, ) + tuple(layer_sizes)
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            print('decoder add fc:{}'.format((in_size, out_size)))

        if dataset_name == 'mnist':
            self.logits = nn.Linear(layer_sizes[-1], output_size)
        elif dataset_name == 'gmm':
            self.means = nn.Linear(layer_sizes[-1], output_size)
            self.log_var = nn.Linear(layer_sizes[-1], output_size)
        else:
            raise NotImplementedError
            # self.means = nn.Linear(layer_sizes[-1], latent_size)
            # self.log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, z):
        h = self.MLP(torch.cat((x, z), dim=-1))
        if self._dataset_name == 'mnist':
            logits = self.logits(h)
            return logits
        elif self._dataset_name == 'gmm':
            means = self.means(h)
            log_var = self.log_var(h)
            return means, log_var
        else:
            raise NotImplementedError

