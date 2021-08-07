""" dmt/data/loading/patch_loader.py

A patch-loader similar to the one in torchio except the queue is continuously
being filled (as opposed to only being filled when it's empty). 
"""

import os
import psutil
import time
import warnings
import torch
import torch.multiprocessing as torch_mp
import multiprocessing as python_mp

from dmt.utils.parse import parse_bool, parse_int, parse_nonnegative_int
from dmt.data.loading.collate import default_collate_fn, return_first


def data_loading_worker(torch_loader, batch_queue):
    pid = os.getpid()
    this_process = psutil.Process(pid)
    
    start = time.time()
    for i, batch in enumerate(torch_loader):
        print(f'Putting batch {i+1} in Q ({time.time() - start:.2f} sec).')
        batch_queue.put(batch)
        mem_usage = this_process.memory_info()[0]/2.**30
        print(f'Using {mem_usage:.2f} GB in PID {pid}')
        start = time.time()
    
    del torch_loader
    print('DaemonLoader loading complete.')
    while True:  # wait for process to be killed
        time.sleep(1)


class PatchLoader:
    
    def __init__(
            self,
            dataset,
            sampler,
            samples_per_volume,
            batch_size,
            collate_fn=default_collate_fn,
            shuffle_samples=True,
            shuffle_patches=True,
            num_workers=3,  # num of workers to load volumes & do transforms
            headstart=False,  # start loading immediately after loader init
            patch_queue_maxsize=64,
            drop_last=True
            ):
        
        if python_mp.get_start_method() != 'fork':
            msg = ('With multiprocess dataloading, you should use the "fork" '
                   'start method for faster worker initialization times and ' 
                   'reduced memory usage (assuming you are only reading from '
                   'dataset components & not modifying them).')
            warnings.warn(msg)
        
        # Multi-processing to load images faster
        self.dataset = dataset
        self.num_workers = parse_nonnegative_int(num_workers, 'num_workers')
        self.shuffle_samples = parse_bool(shuffle_samples, 'shuffle_samples')
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=self.num_workers,
            collate_fn=return_first, shuffle=self.shuffle_samples
        )
        
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.batch_size = parse_nonnegative_int(batch_size, 'batch_size')
        self.samples_per_volume = parse_nonnegative_int(samples_per_volume,
                                                        'samples_per_volume')
        self.shuffle_patches = parse_bool(shuffle_patches, 'shuffle_patches')
        self.drop_last = parse_bool(drop_last, 'drop_last')
        
        self.patch_queue_maxsize = parse_int(patch_queue_maxsize, 
                                             'patch_queue_maxsize')
        self._patch_queue = torch_mp.Queue(maxsize=self.patch_queue_maxsize)
        self._daemon_loader = None
        self._batch_count = None
        self._loading = False
        
        self._samples_iterable = None
        if headstart:
            self.start_loading()
    
    @property
    def num_samples(self):
        return len(self.dataset)
    
    @property
    def num_batches(self):
        if self.drop_last:
            return self.patches_per_epoch // self.batch_size
        mod = self.patches_per_epoch % self.batch_size
        return self.patches_per_epoch // self.batch_size + int(mod > 0)
    
    @property
    def patches_per_epoch(self):
        return self.num_samples * self.samples_per_volume
    
    @property
    def is_queue_empty(self):
        return self.patch_queue.empty()
    
    @property
    def is_queue_full(self):
        return self.patch_queue.full()
            
    @property
    def worker_pid(self):
        if self.daemon_loader is not None:
            return self.daemon_loader.pid
        return None
    
    @property
    def worker_memory(self):
        if self.daemon_loader is not None:
            process = psutil.Process(self.worker_pid)
            mem_usage_mb = process.memory_info().rss / (1024 ** 2)
            return mem_usage_mb
        return None
        
    @property
    def main_pid(self):
        return os.getpid()
    
    def __len__(self):
        return self.num_batches
    
    
    ### ------ #     Main API     # ----- ###
    
    def __iter__(self):
        self.batch_count = 0
        if self._samples_iterable is None:
            self._reset_samples_iterable()
        return self
    
    def __next__(self):
        if self.batch_count >= self.num_batches:
            self._samples_iterable = None
            raise StopIteration
        
    
    
    ### ------ #   Sample Loading  # ----- ###
    
    def _reset_samples_iterable(self):
        """ """
        pass
    
    def _get_next_sample(self):
        try:
            sample = next(self._samples_iterable)
        except StopIteration as e:
            self._reset_samples_iterable()
            sample = next(self._samples_iterable)
        return sample
    
    
    ### ------ #   Continuous Patch Loading & Multiprocessing  # ----- ###
    
    def start_loading(self):
        if self.daemon_loader != None and self.loading:
            print(f'Already loading.')
            return
        
        self.daemon_loader = torch.multiprocessing.Process(
            target=loading_worker, args=(self.torch_loader, self.batch_queue))
        self.daemon_loader.start()
        self.loading = True
    
    def stop_loading(self):
        if self.daemon_loader is None:
            return
        self.daemon_loader.kill()
        self.daemon_loader = None
        self.loading = False
        
    def reset_loading(self):
        self.stop_loading()
        
        # clear queue
        from queue import Empty
        try:
            while True:
                self.batch_queue.get_nowait()
        except Empty:
                return
    
    def __len__(self):
        return len(self.torch_loader)
    
    def __iter__(self):
        if not self.loading:
            self.reset_loading()
            self.start_loading()
        self.batch_count = 0
        return self
    
    def __next__(self):
        if self.batch_count < len(self):
            batch = self.batch_queue.get()
            self.batch_count += 1
            return batch
        else:
            self.reset_loading()
            raise StopIteration
        
    def __del__(self):
        self.reset_loading()

