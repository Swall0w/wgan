import chainer 
import chainer.function as F
import chainer.links as L

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator,self).__init__(
            fc1 = L.Linear(None, 800),
            fc2 = L.Linear(None, 28*28)
        )
    def __call__(self,z,test=False):
        h = F.relu(self.fc1(z))
        y = F.reshape(F.sigmoid(self.fc2(h)),(-1,1,28,28))
        return y
        
