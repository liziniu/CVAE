import time
import csv
import os

import numpy as np
import torch

from cvae import CVAE, gaussian_log_density
import pypr.clustering.gmm as gmm


INPUT_SIZE = 1
OUTPUT_SIZE = 1
LATENT_SIZE = 3
BATCH_SIZE = 64
TOTAL_SAMPLES = int(5e3)
N_TRAINS = 0.8
N_INFERENCE = 10
SEED = 2020
ALPHA = .1

np.random.seed(SEED)
torch.manual_seed(SEED)


def generate_data(n_samples):
    mc = [0.4, 0.4, 0.2]  # Mixing coefficients
    centroids = [
        np.array([0, 0]),
        np.array([3, 3]),
        np.array([0, 4])
    ]
    ccov = [
        np.array([[1, 0.4], [0.4, 1]]),
        np.diag((1, 2)),
        np.diag((0.4, 0.1))
    ]

    # Generate samples from the gaussian mixture model
    samples = gmm.sample_gaussian_mixture(centroids, ccov, mc, samples=n_samples)
    xs, ys = samples[:, 0], samples[:, 1]
    probs = np.zeros([n_samples], dtype=np.float32)
    for it in range(n_samples):
        input_ = np.array([xs[it], np.nan])
        con_cen, con_cov, new_p_k = gmm.cond_dist(input_, centroids, ccov, mc)
        prob = gmm.gmm_pdf(ys[it], con_cen, con_cov, new_p_k)
        probs[it] = prob
    return xs, ys, probs

xs, ys, probs = generate_data(TOTAL_SAMPLES)
nb_train = int(TOTAL_SAMPLES * N_TRAINS)
xs_train, ys_train, probs_train = xs[:nb_train], ys[:nb_train], probs[:nb_train]
xs_test, ys_test, probs_test = xs[nb_train:], ys[nb_train:], probs[nb_train:]

cvae = CVAE(
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    latent_size=LATENT_SIZE,
    encoder_layer_sizes=(64, 64),
    decoder_layer_sizes=(64, 64),
    dataset_name='gmm',
    alpha=ALPHA,
)

time_st = time.time()
os.makedirs('out/gmm', exist_ok=True)
save_dir = 'out/gmm/%d_%d_%d_%d_%.2f' % (INPUT_SIZE, OUTPUT_SIZE, LATENT_SIZE, N_INFERENCE, ALPHA)
os.makedirs(save_dir, exist_ok=True)
file = open(os.path.join(save_dir, 'progress.csv'), 'wt')
csv_writer = None

for it in range(int(2e4)):
    inds = np.random.choice(nb_train, BATCH_SIZE, replace=False)
    x, y, prob = xs_train[inds], ys_train[inds], probs_train[inds]
    loss = cvae.update(x[:, None], y[:, None])

    if it % int(1e3) == 0:
        inds = np.random.choice(nb_train, BATCH_SIZE, replace=False)
        x, y, prob = xs_train[inds], ys_train[inds], probs_train[inds]
        prob_pred = []
        for _ in range(N_INFERENCE):
            mean, log_var = cvae.sample(x[:, None])
            _log_prob = gaussian_log_density(mean, log_var, torch.tensor(y[:, None], dtype=torch.float32))
            _prob = torch.exp(_log_prob).detach().numpy()
            prob_pred.append(_prob)
        prob_pred = np.mean(np.asarray(prob_pred), axis=0)
        test_loss = np.mean((prob - prob_pred)**2)
        print('Iter-{}; Train Loss: {:.4f}, Test Loss:{:.4f} fps:{}'.format(it, loss.data, test_loss,
                                                                            (it+1) // (time.time() - time_st)))
        print(np.round(prob[:3], 3), np.round(prob_pred[:3], 3))

        logs = dict(
            step=it,
            train_loss=loss.data.numpy(),
            test_loss=test_loss,
        )
        if csv_writer is None:
            csv_writer = csv.DictWriter(file, fieldnames=logs.keys())
            csv_writer.writeheader()
        csv_writer.writerow(logs)
        file.flush()
