
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



def print_processes():
    import psutil, os
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    print(f'Current process {os.getpid()} has {len(children)} child processes.')
    for i, child in enumerate(children):
        name = child.name()
        child_children = child.children()
        mem = child.memory_info().rss / 2. ** 30
        print(f'   Child [{i+1}] pid={child.pid} name={name} ({len(child_children)} kids '
              f', running={child.is_running()}) parent={child.ppid()} mem={mem:.2f}GB')


def load(sample_set, num_workers=4, transforms=None):
    
    dataset = Dataset(sample_set, transforms=transforms)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=6, 
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True
    )
    time.sleep(2)
            
    batch_start = time.time()
    print(f' Starting loader iteration.')
    for i, batch in enumerate(loader):
        if i in (2, 10):
            print(f'Batch iteration 2. Processes:')
            print_processes()
        time.sleep(1)
        t = time.time() - batch_start
        print(f'\n  Batch {i+1} RECEIVED! ({len(batch)} items, {t:.2f} sec)')
        for k in batch.keys():
            print(f'  {k}: ', end='')
            for t in batch[k]:
                print(f'{t.shape}', end='')
        print()
        batch_start = time.time()


