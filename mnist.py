import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from cvae import CVAE


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


INPUT_SIZE = mnist.train.labels.shape[1]
OUTPUT_SIZE = mnist.train.images.shape[1]
LATENT_SIZE = 100
BATCH_SIZE = 64

cvae = CVAE(
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    latent_size=LATENT_SIZE,
    encoder_layer_sizes=(200, 200),
    decoder_layer_sizes=(200, 200),
    dataset_name='mnist'
)

time_st = time.time()

for it in range(int(2e4)):
    y, x = mnist.train.next_batch(BATCH_SIZE)

    loss = cvae.update(x, y)
    if it % int(1e3) == 0:
        print('Iter-{}; Loss: {:.4f}, fps:{}'.format(it, loss.data, (it+1) // (time.time() - time_st)))

        x = np.zeros(shape=[BATCH_SIZE, INPUT_SIZE], dtype=np.float32)
        x[:, np.random.randint(0, 10)] = 1.
        samples = cvae.sample(x).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/mnist'):
            os.makedirs('out/mnist')

        plt.savefig('out/mnist/{}.png'.format(str(it).zfill(3)), bbox_inches='tight')
        plt.close(fig)
