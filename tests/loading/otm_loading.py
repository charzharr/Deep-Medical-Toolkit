
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
TRANSFORMS = torchio.Compose([
    torchio.RandomFlip(),
    torchio.RandomBlur(),
    torchio.RandomNoise(),
])


ExTup = namedtuple('ExTup', 'image id')

def sample_processing_fn(samples):
    # get single sample
    sample = samples[0]
    # tens = sample.image.tensor
    
    # crops
    num_crops = NUM_CROPS
    subject = torchio.Subject(
        image=torchio.ScalarImage(sample.image.path),
        mask=torchio.LabelMap(sample.label.path),
        id=sample.id, 
        name=sample.name)
    crop_iter = SAMPLER(subject)
    
    # data augmentation on crops
    examples = list(itertools.islice(crop_iter, num_crops))
    examples = [TRANSFORMS(e) for e in examples]  # tio subjects
    examples = [ExTup(s.image.data, s.id) for s in examples]
    # examples = [{'image': s.image.data, 'id': s.id} for s in examples]
    return examples 


def collate_fn(batch_subjects):
    batch_d = {'X': [], 'ids': []}
    for subject in batch_subjects:
        batch_d['X'].append(subject.image)
        batch_d['ids'].append(subject.id)
    return batch_d

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

def load(sample_set, num_workers=4, batch_size=16):
    num_workers = 3
    loader = OneToManyLoader(
                sample_set,
                sample_processing_fn,
                examples_per_sample=NUM_CROPS,
                example_collate_fn=collate_fn,
                batch_size=16,
                shuffle_samples=True,
                shuffle_patches=True,
                num_workers=num_workers,
                headstart=False,
                drop_last=True,
                example_queue_maxsize=800
                )
    
    print(f'Preload processes: ')
    print_processes()

    for _ in range(4):
        print(f'Started loader loading.')
        loader.start_loading()
        time.sleep(2)
        # import IPython; IPython.embed(); 
        
        epoch_start = batch_start = time.time()
        print(f' Starting loader iteration.')
        for i, batch in enumerate(loader):
            if i in (2, 10):
                print(f'Batch iteration 2. Processes:')
                print_processes()
            t = time.time() - batch_start
            print(f'\n  Batch {i+1} RECEIVED! ({len(batch)} items, {t:.2f} sec)')
            print(f"IDs: {batch['ids']}")
            print(f'  Tensors (X): ', end='')
            for t in batch['X']:
                print(f'{t.shape}', end='')
            print()
            batch_start = time.time()
        
        print(f'Inner epoch took {time.time() - epoch_start:.2f} sec')
        print(f'End of loader epoch.. sleeping for 10.')
        time.sleep(5)
        # print_processes()
        # import IPython; IPython.embed(); 
            