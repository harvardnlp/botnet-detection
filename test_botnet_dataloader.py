from tqdm import tqdm

from botdet.data.dataset_botnet import BotnetDataset
from botdet.data.dataloader import GraphDataLoader


if __name__ == '__main__':
    dataset = BotnetDataset(name='chord', split='train', graph_format='pyg', in_memory=True)
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(dataset)
    for batch in tqdm(loader):
        pass
