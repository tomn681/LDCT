from dataset import DefaultDataset

dataset = DefaultDataset('../DefaultDataset')

print(f'Mean: {dataset.mean}\nStd: {dataset.std}')
