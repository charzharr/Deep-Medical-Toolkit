""" dmt/data/loading/oto_loader.py
OTO = OneToOne. Meaning 1 subject loads exactly 1 batch example. 

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
import logging, warnings
import multiprocessing
import torch


def batch_loading_worker(torch_loader, batch_queue):
    
    pid = os.getpid()
    this_process = psutil.Process(pid)
    
    start = time.time()
    for i, batch in enumerate(torch_loader):
        # print(f'Putting batch {i+1} in Q ({time.time() - start:.2f} sec).')
        batch_queue.put(batch)
        mem_usage = this_process.memory_info()[0]/2.**30
        logging.info(f'OTOWorker: Using {mem_usage:.2f} GB in PID {pid}')
        start = time.time()
    
    del torch_loader
    print('OTOLoader loading complete.')
    while True:  # wait for process to be killed
        time.sleep(1)


class OneToOneLoader:
    
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
        
        self.batch_loader = None
        self.loading = False
        self.batch_count = None
        self.iter_runs = 0
        
        if headstart:
            self.start_loading()
    
    @property
    def main_pid(self):
        return os.getpid()
    
    @property
    def worker_pid(self):
        if self.batch_loader is not None:
            return self.batch_loader.pid
        return None
    
    @property
    def worker_memory(self):
        if self.batch_loader is not None:
            process = psutil.Process(self.worker_pid)
            mem_usage_mb = process.memory_info().rss / (1024 ** 2)
            return mem_usage_mb
        return None
    
    def start_loading(self):
        if self.batch_loader != None and self.loading and self.iter_runs > 0:
            warnings.warn(f'(start_loading) Loader is already loading.')
            return
        
        logging.info('OTOLoader: Starting process to load batches')
        # print('OTOLoader: Starting process to load batches')
        args = (self.torch_loader, self.batch_queue)
        self.batch_loader = torch.multiprocessing.Process(
            target=batch_loading_worker, args=args)
        self.batch_loader.start()
        self.loading = True
    
    def stop_loading(self):
        self.loading = False
        if self.batch_loader is None:
            # warnings.warn(f'(stop_loading) Loader process not instantiated.')
            return
        if self.batch_loader.is_alive():
            logging.info('OTOLoader: Killing loader process to load batches.')
            # print('OTOLoader: Killing loader process to load batches.')
            self.batch_loader.kill()
            self.batch_loader.join()
            assert not self.batch_loader.is_alive()
        self.batch_loader = None
        logging.info('OTOLoader: Successfully killed loader process.')
        # print('OTOLoader: Successfully killed loader process.')
        self.iter_runs += 1
        
    def stop_loading_and_reset(self):
        """ Same as stop_loading except, it also clears the batch_queue. """
        start = time.time()
        self.stop_loading()
        
        logging.info('OTOLoader: Clearing batch queue.')
        from queue import Empty
        try:
            while True:  # clear queue
                self.batch_queue.get_nowait()
                print(f'OTOLoader deleted batch element. Qempty: '
                      f'{self.batch_queue.empty()}')  # this shouldn't print
        except Exception as e:
            if isinstance(e, Empty):
                logging.info('OTOLoader: Successfully cleared batch queue.')
            elif not self.batch_queue.empty():
                raise e
        elapsed = time.time() - start
        logging.info(f'OTOloader: took {elapsed:.2f} sec to kill loader.')
        return
    
    def __len__(self):
        return len(self.torch_loader)
    
    def __iter__(self):
        if not self.loading:
            self.stop_loading_and_reset()
            self.start_loading()
        self.batch_count = 0
        return self
    
    def __next__(self):
        if self.batch_count < len(self):
            batch = self.batch_queue.get()
            self.batch_count += 1
            return batch
        else:
            self.stop_loading_and_reset()
            raise StopIteration
        
    # def __del__(self):
    #     logging.info('OTOLoader: Deleting instance & components.')
    #     self.stop_loading_and_reset()
    #     del self.torch_loader
    #     self.torch_loader = None
    #     del self.batch_queue 
    #     self.batch_queue = None
        

