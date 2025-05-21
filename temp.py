from config import config
from utils.dataset import DefaultDataset


dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices, train=False)
print(len(dataset))
