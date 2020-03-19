#!/usr/bin/env python

import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda
from chainer import serializers
import numpy as np
from chainer.datasets import tuple_dataset
import net
import net_img
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    if 0 <= args.gpu:
        cuda.get_device_from_id(args.gpu).use()

    # net内VAEオブジェクトの生成
    textVae = net.VAE(600, args.dimz,300,100)
    chainer.serializers.load_npz("birds_txt.npz", textVae)
    if 0 <= args.gpu:
        textVae.to_gpu()  # GPUを使うための処理

    model = net_img.VAE(1, 20, 64, textVae)
    chainer.serializers.load_npz("birds_all.npz", model)
    if 0 <= args.gpu:
        model.to_gpu()  # GPUを使うための処理
    # optimizer(パラメータ更新用)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # モデルの読み込み npzはnumpy用
    """
    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, model)
"""

    traint = np.load('birds_txt.npy')
    
    traini = np.load('birds_img.npy')
    
    traini = traini.reshape((len(traini), 1, 128, 128))
    
    train = tuple_dataset.TupleDataset(traint, traini)

    train, test = train_test_split(train, test_size=0.2, random_state=50)
    
    


#------------------イテレーターによるデータセットの設定-----------------------------------
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
#---------------------------------------------------------------
    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=model.get_loss_func())

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        eval_func=model.get_loss_func(k=10)))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/rec_loss', 'validation/main/rec_loss', 'elapsed_time']))
    # trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # トレーナーの実行
    #trainer.run()

    # Visualize the results
    def save_images(x, filename):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
        for ai, xi in zip(ax.flatten(), x):
            ai.imshow(xi.reshape(128, 128), cmap=cm.gray)
        fig.savefig(filename)
    

    #model.to_cpu()

    train_ind = [4]
    x = chainer.Variable(np.asarray(traint[train_ind]))
    print(x)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        mu, ln_var = textVae.encode(x)
        x2 = model.decode(mu)
        print(x2)
        
    save_images(x2.array, os.path.join(args.out, 'train'))
    save_images(x2.array, os.path.join(args.out, 'train_reconstructed'))

"""
    x = chainer.Variable(traint[0])
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    #save_images(x.array, os.path.join(args.out, 'test'))
    save_images(x1.array, os.path.join(args.out, 'test_reconstructed'))
    """
"""
    # draw images from randomly sampled z
    z = chainer.Variable(
        np.random.normal(0, 1, (9, args.dimz)).astype(np.float32))
    x = model.decode(z)
    save_images(x.array, os.path.join(args.out, 'sampled'))

"""
if __name__ == '__main__':
    main()
    
