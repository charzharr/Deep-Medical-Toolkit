""" dmt/data/loading/daemon_loader.py

Daemon loader wraps torch's Dataloader by continuously loading into a queue.
'Daemon' here refers to its original software interpretation (not Python's)
where a process works continuously in the background.

Added Functionality:
  - Allows asynchronous loading from an additional single worker.
  - Adds headstart functionality so there's no need to wait for 1st batch & init
    - allows preloading of training set 
    - allows validation / test sets to load near end of training epoch
  - Wraps additional process information like memory & cpu usage
"""

import os
import psutil
import time
import warnings
import multiprocessing
import torch


def batch_loading_worker(torch_loader, batch_queue):
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


class DaemonLoader:
    
    def __init__(
            self,
            *args, 
            headstart=False, 
            queue_maxsize=32,
            **kwargs):
        
        if multiprocessing.get_start_method() != 'fork':
            msg = ('With multiprocess dataloading, you should use the "fork" '
                   'start method for faster worker initialization times and ' 
                   'reduced memory usage (assuming you are only reading from '
                   'dataset components & not modifying them).')
            warnings.warn(msg)
        
        self.torch_loader = torch.utils.data.DataLoader(*args, **kwargs)
        self.batch_queue = torch.multiprocessing.Queue(maxsize=queue_maxsize)
        
        self.daemon_loader = None
        self.loading = False
        self.batch_count = None
        
        if headstart:
            self.start_loading()
            
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
    
    def start_loading(self):
        if self.daemon_loader != None and self.loading:
            print(f'Already loading.')
            return
        
        args = (self.torch_loader, self.batch_queue)
        self.daemon_loader = torch.multiprocessing.Process(
            target=batch_loading_worker, args=args)
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
        

