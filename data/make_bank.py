import sys
sys.path.append('../')
import argparse
from tqdm import tqdm
import torch
from data import Data
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('func', type=str)
parser.add_argument('splitting', type=int)

def make_bank(splitting):
    if splitting == 1:
        pre = './processed/lin/'
        conf = Config(11, 1)
    elif splitting == 2:
        pre = './processed/ji/'
        conf = Config(11, 2)
    elif splitting == 3:
        pre = './processed/l/'
        conf = Config(4, 3)
    d = Data(True, conf)
    senselist = []
    idxlist = []
    for idx, a1, a2, sense, conn in tqdm(d.train_loader, total=d.train_size//conf.batch_size+1):
        senselist.append(sense)
        idxlist.append(idx)
    idxlist = torch.cat(idxlist, 0)
    senselist = torch.cat(senselist, 0)
    assert len(idxlist) == len(senselist)
    new_senselist = torch.LongTensor(senselist.size())
    for i in range(len(idxlist)):
        new_senselist[idxlist[i]] = senselist[i]
    torch.save(new_senselist, pre+'bank.pkl')

def test_bank(splitting):
    if splitting == 1:
        pre = './processed/lin/'
    elif splitting == 2:
        pre = './processed/ji/'
    elif splitting == 3:
        pre = './processed/l/'
    bank = torch.load(pre+'bank.pkl')
    print(bank)

if __name__ == "__main__":
    A = parser.parse_args()
    if A.func == 'pre':
        if A.splitting == 1:
            make_bank(1)
        elif A.splitting == 2:
            make_bank(2)
        elif A.splitting == 3:
            make_bank(3)
    elif A.func == 'test':
        if A.splitting == 1:
            test_bank(1)
        elif A.splitting == 2:
            test_bank(2)
        elif A.splitting == 3:
            test_bank(3)
    else:
        raise Exception('wrong args')
