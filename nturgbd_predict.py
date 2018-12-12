import argparse
import glob
import os

import numpy as np
from PIL import Image
from chainer import Variable

from net import Discriminator
from net import Encoder
from net import Decoder
from nturgbd_dataset import NTURGBDDatasetForSkeleton2Flow, NTURGBDDatasetForSkeleton2RGB

datasets = {
    "skeleton2rgb": NTURGBDDatasetForSkeleton2RGB,
    "skeleton2flow": NTURGBDDatasetForSkeleton2Flow
}


def save_image(y, flowx_path, flowy_path, mode=None):
    C, H, W = y.shape
    flowx_im = y[0]
    flowy_im = y[1]
    dn = flowx_path.rsplit("/", 1)[0]
    os.makedirs(dn, exist_ok=True)
    Image.fromarray(flowx_im, mode=mode).convert('L').save(flowx_path)
    Image.fromarray(flowy_im, mode=mode).convert('L').save(flowy_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-i', '--input-dir', type=str,
        help='Directory including joint and edge images')
    parser.add_argument(
        '-o', '--output-dir', type=str,
        help='Output directory of estimated flow images')
    parser.add_argument(
        '-sc', '--c', type=str, default="view", choices=["view", "subject"],
        help='Choice of split criterion')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument(
        '--dataset-name', '-d', type=str, default="skeleton2flow", choices=datasets.keys(),
        help='Number of images in each mini-batch')
    parser.add_argument(
        '--enc-model', '-em', type=str, help='Path to encoder model file')
    parser.add_argument(
        '--dec-model', '-dm', type=str, help='Path to decoder model file')
    args = parser.parse_args()

    # Setup
    dataset = datasets[args.dataset_name]
    enc = Encoder(in_ch=dataset.in_ch)
    dec = Decoder(out_ch=dataset.out_ch)

    # Load models
    chainer.serializers.load_npz(args.enc_model, enc)
    chainer.serializers.load_npz(args.dec_model, dec)

    # Search edge and joint images
    train = dataset(
        subset="train", dataDir=args.input_dir, split_criterion=args.split_criterion)
    test = dataset(
        subset="test", dataDir=args.input_dir, split_criterion=args.split_criterion)

    # Generate flow images for training data
    input_images = list(zip(train.edge_paths, train.joint_paths))
    output_images = []
    for edge_p, joint_p in input_images:
        flowx_p = edge_p.replace("edge_", "flowx_").replace(args.input_dir, args.output_dir)
        flowy_p = flowx_p.replace("flowx_", "flowy_")
        output_images.append((flowx_p, flowy_p))

    for i in range(len(train)):
        x, _ = train.get_example(i)
        x = Variable(x[np.newaxis, ...])
        y = dec(enc(x)).get()[0]
        y = np.asarray(np.clip(y * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(y, *output_images[i])

    # Generate flow images for test data
    input_images = list(zip(test.edge_paths, test.joint_paths))
    output_images = []
    for edge_p, joint_p in input_images:
        flowx_p = edge_p.replace("edge_", "flowx_").replace(args.input_dir, args.output_dir)
        flowy_p = flowx_p.replace("flowx_", "flowy_")
        output_images.append((flowx_p, flowy_p))

    for i in range(len(test)):
        x, _ = test.get_example(i)
        x = Variable(x[np.newaxis, ...])
        y = dec(enc(x)).get()[0]
        y = np.asarray(np.clip(y * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(y, *output_images[i])
