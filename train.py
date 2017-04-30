import argparse
import os
import numpy as np
from PIL import Image
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

def out_gen_image(generator, H, W, rows, cols, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols
        xp = generator.xp
        z = xp.random.randn(n_images, 1, int(H/4), int(W/4)).astype(xp.float32)
        x = generator(z, test = True)
        x = chainer.cuda.to_cpu(x.data)

        x = np.asarray(np.clip(x * 255, 0.0, 255.0),dtype=np.uint8)
        channels = x.shape[1]
        x = x.reshape((rows, cols, channels, H, W))
        x = x.transpose(0,3,1,4,2)
        x = x.reshape((rows * H, cols * W, channels))
        x = np.squeeze(x)

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>5}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image

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

    trainer.extend(extensions.dump_graph('wasserstein distance'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch,'epoch'))
    trainer.extend(extensions.LogReport())
#    trainer.extend(extensions.PlotReport(['wasserstein distance'],'epoch', file_name = 'distance.png'))
#    trainer.extend(extensions.PlotReport(['epoch','wasserstein distance','loss/generator','elapsed_time']))
    trainer.extend(extensions.PrintReport())
    trainer.extend(extensions.ProgressBar())
    trainer.extend(out_gen_image(generator,28,28,5,5,args.output),trigger=(1,'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
