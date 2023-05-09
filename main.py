from options import args
from utils.dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset(args)
    dataset()