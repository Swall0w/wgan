import argparse
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

if __name__ == '__main__':
    main()
