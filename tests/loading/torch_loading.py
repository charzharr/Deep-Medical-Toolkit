
import torch
import time


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, sample_set, transforms=None):
        self.sample_set = sample_set
        self.transforms = transforms
    
    def __len__(self):
        return len(self.sample_set)
    
    def __getitem__(self, idx):
        sample = self.sample_set[idx]
        image = sample.image.array
        label = sample.label.array
        return {
            'image': image,
            'label': label
        }
        
def collate_fn(batch):
    batch_d = {}
    for k in batch[0].keys():
        batch_d[k] = [torch.tensor(b[k]) for b in batch]
    return batch_d


def loading_fn(dataset, num_workers, batch_queue):
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=6, 
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True
    )
    for i, batch in enumerate(loader):
        print(f'Putting batch {i+1} in queue'); start = time.time()
        batch_queue.put(batch)
        print(f'Putting batch {i+1} complete ({time.time() - start:.2f} sec).')

# def load(sample_set, num_workers=4):
#     dataset = Dataset(sample_set)
    
#     import multiprocessing
#     batch_queue = torch.multiprocessing.Queue()
#     args = (dataset, num_workers, batch_queue)
#     process = torch.multiprocessing.Process(target=loading_fn, args=args)
#     process.start()
    
#     time.sleep(5)
#     print(f'WOKE')
    
#     batch_start = time.time()
#     counter = 0
#     while counter < 16:
#         batch = batch_queue.get()
#         t = time.time() - batch_start
#         print(f'\n  Batch {counter+1} RECEIVED! ({len(batch)} items, {t:.2f} sec)')
#         for k in batch.keys():
#             print(f'  {k}: ', end='')
#             for t in batch[k]:
#                 print(f'{t.shape}', end='')
#         print()
#         batch_start = time.time()
#         counter += 1
#     print('quitting process')


def load(sample_set, num_workers=4):
    
    dataset = Dataset(sample_set)
    from torch.utils.data import DataLoader
    from dmt.data.loading.daemon_loader import DaemonLoader
    loader = DaemonLoader(
        dataset,
        headstart=True,
        num_workers=num_workers,
        batch_size=6, 
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True
    )
    
    time.sleep(5)
    
    print(loader.main_pid, loader.worker_pid, loader.worker_memory)
    
    batch_start = time.time()
    print(f' Starting loader iteration.')
    for i, batch in enumerate(loader):
        time.sleep(1)
        t = time.time() - batch_start
        print(f'\n  Batch {i+1} RECEIVED! ({len(batch)} items, {t:.2f} sec)')
        for k in batch.keys():
            print(f'  {k}: ', end='')
            for t in batch[k]:
                print(f'{t.shape}', end='')
        print()
        batch_start = time.time()