import argparse
import chainer
from chainer import optimizers, training
from chainer.training import extensions
from chainer.dataset import iterator
from chainer.dataset import convert
from model import Generator, Critic
from util import WGANUpdater, WeightClipping
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch','-b',type=int,default=100,help='number of minibatch')
    parser.add_argument('--epoch','-e',type=int,default=100,help='number of epoch')
    parser.add_argument('--gpu','-g',type=int,default=-1,help='number of gpu')
    parser.add_argument('--output','-o',default='result',help='output directory')
    parser.add_argument('--resume','-r',default='',help='resume the training from snapshot')
    parser.add_argument('--unit','-u',type=int,default=500,help='number of unit')
    return parser.parse_args()

def main():
    args = arg()
    generator = Generator()
    critic = Critic()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()

    op_g = optimizers.RMSprop(5e-5)
    op_g.setup(generator)
    op_g.add_hook(chainer.optimizer.GradientClipping(1))

    op_c = optimizers.RMSprop(5e-5)
    op_c.setup(critic)
    op_c.add_hook(chainer.optimizer.GradientClipping(1))
    op_c.add_hook(WeightClipping(0.01))

    train, test = chainer.datasets.get_mnist(ndim=3,withlabel=False)
    train_iter = chainer.iterators.SerialIterator(train, args.batch)

    updater = WGANUpdater(train_iter, generator, critic, 5, op_g, op_c, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch,'epoch'),out=args.output)


if __name__ == '__main__':
    main()
