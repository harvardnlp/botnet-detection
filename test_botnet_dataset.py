from botdet.data.dataset_botnet import BotnetDataset


if __name__ == '__main__':
    dataset = BotnetDataset(name='chord', split='train', graph_format='pyg')
    print(dataset)
    print(len(dataset))
    print(dataset[0])
    breakpoint()
