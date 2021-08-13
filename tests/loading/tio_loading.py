
from collections import namedtuple
import os
import sys, pathlib
import psutil
import torch
import torchio
import time
import itertools

curr_path = pathlib.Path(__file__).absolute()
sys.path.append(str(curr_path.parent.parent.parent))

from dmt.data.loading.otm_loader import OneToManyLoader


NUM_CROPS = 3
SAMPLER = torchio.UniformSampler(patch_size=32)


def print_processes():
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    print(f'Current process {os.getpid()} has {len(children)} child processes.')
    for i, child in enumerate(children):
        name = child.name()
        child_children = child.children()
        mem = child.memory_info().rss / 2. ** 30
        print(f'   Child [{i+1}] pid={child.pid} name={name} ({len(child_children)} kids '
              f', running={child.is_running()}) parent={child.ppid()} mem={mem:.2f}GB')

def load(subject_set, num_workers=4, batch_size=16):
    num_workers = 3
    
    Qset = torchio.Queue(
        subject_set,
        30,
        3,
        SAMPLER,
        num_workers=num_workers
    )
    loader = torch.utils.data.DataLoader(Qset, batch_size=batch_size)
    
    print(f'Preload processes: ')
    print_processes()

    for _ in range(4):
        print(f'Started loader loading.')
        
        epoch_start = batch_start = time.time()
        print(f' Starting loader iteration.')
        for i, batch in enumerate(loader):
            if i in (2, 10):
                print(f'Batch iteration 2. Processes:')
                print_processes()
            t = time.time() - batch_start
            images = batch['image']['data']
            ids = batch['id']
            print(f'\n  Batch {i+1} RECEIVED! ({images.shape[0]} items, {t:.2f} sec)')
            print(f"IDs: {ids}")
            print(f'  Tensor (X): {images.shape}', end='')
            print()
            batch_start = time.time()
        
        print(f'Inner epoch took {time.time() - epoch_start:.2f} sec')
        print(f'End of loader epoch.. sleeping for 10.')
        time.sleep(5)
        # print_processes()
        # import IPython; IPython.embed(); 
            