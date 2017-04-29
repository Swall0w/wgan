import chainer
from chainer import training
from chainer.dataset import iterator as itr_module
from chainer.dataset import convert
class WGANUpdater(training.StandardUpdater):
    def __init__(self,iterator, generator,critic, num, op_g, op_c, device):
        if isinstance(iterator, itr_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.generator = generator
        self.critic = critic
        self.num = num
        self._optimizers = {'generator':op_g,'critic':op_c}
        self.device = device
        self.converter = convert.concat_examples
        self.itrtn = 0

    def update_core(self):
        batch = self._itrs['main'].next()
        images = self.converter(batch, self.device)
        batchsize = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        xp = chainer.cuda.get_array_module(images)

        z = xp.random.normal(size=(batchsize, 1, H/4, W/4).astype(xp.float32))
        generated = self.generator(z)

        real_y = self.critic(images)
        fake_y = self.critic(generated)

        was_distance = real_y - fake_y
        loss_critic = - was_distance
        loss_generator = -fake_y

        self.critic.cleargrads()
        loss_critic.backward()
        self._optimizers['critic'].update()

        if self.itrtn < 2500 and self.itrtn % 100 == 0:
            self.generator.cleargrads()
            loss_generator.backward()
            self._optimizers['generator'].update()

        if self.itrtn > 2500 and self.itrtn % self.num == 0:
            self.generator.cleargrads()
            loss_generator.backward()
            self._optimizers['generator'].update()

        chainer.reporter.report({
            'loss/generator':loss_generator,
            'loss/critic':loss_critic,
            'wasserstein distance': was_distance
            })

class WeightClipping():
    name = 'WeightClipping'
    def __init__(self,threshold):
        self.threshold = threshold
    
    def __call__(self, opt):
        for param in opt.target.params():
            xp = chainer.cuda.get_array_module(param.data)
            param.data = xp.clip(param.data,-self.threshold, self.threshold)
