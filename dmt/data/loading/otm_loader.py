""" dmt/data/loading/patch_loader.py

A patch-loader similar to the one in torchio except the queue is continuously
being filled (as opposed to only being filled when it's empty). 
"""

import os
import psutil
import time
import logging, warnings
import torch
import torch.multiprocessing as torch_mp
import multiprocessing as python_mp

from dmt.utils.parallel import is_process_running
from dmt.utils.parse import parse_bool, parse_int, parse_nonnegative_int
from dmt.data.loading.collate import default_collate_fn


def example_loading_worker(example_loader, example_queue, shuffle):
    
    examples_storage = set()
    def put_example(ex):
        """ Only called when shuffling is required. """
        if ex is None:
            example_queue.put(examples_storage.pop())
        else:
            examples_storage.add(ex)
            if len(examples_storage) >= 10:
                example_queue.put(examples_storage.pop())
    
    pid = os.getpid()
    ppid = os.getppid()
    this_process = psutil.Process(pid)
    
    start = time.time()
    for i, examples in enumerate(example_loader):
        # print(f'Putting ex {i+1} in Q ({time.time() - start:.2f} sec).')
        # mem_usage = this_process.memory_info()[0]/2.**30
        # logging.debug(f'OTMWorker: Using {mem_usage:.2f} GB in PID {pid}')
        
        if not shuffle:
            for ex in examples:
                example_queue.put(ex)
        else:
            for ex in examples:
                put_example(ex)
        start = time.time()
    
    while examples_storage:
        put_example(None)
    logging.info(f'\n💀 OTMLoader-Side: done loading examples. Exiting now!')


class OneToManyLoader:
    """ Multiprocessing batch loader which creates multiple batch examples from
        a single dataset sample (multiple as in at least one). """
    
    def __init__(
            self,
            dataset,
            sample_processing_fn,  # contains sampler & crop transforms
            examples_per_sample,
            example_collate_fn=default_collate_fn,
            batch_size=16,
            shuffle_samples=True,
            shuffle_patches=True,
            num_workers=3,  # num of workers to load volumes & do transforms
            headstart=False,  # start loading immediately after loader init
            example_queue_maxsize=128,
            drop_last=True
            ):
        """
        Args:
            dataset: Torch-Dataset-like object that collects & preprocesses
                sample data.
            sampler: Samples one or multiple patches given sample data.
            batch_size: number of sampler outputs to collate into a batch.
                Note if sampler outputs 3 patches, batch size = 2 would
                result in a batch with 6 tensors.
            shuffle_samples: Flag to shuffle samples during loading.
            shuffle_patches: Flag to shuffle patches before collation.
            num_workers: Number of processes used for data loading.
            headstart: Flag to start workers right away.
        """
        
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
        
        # Sample processing and example loading
        self.sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=self.num_workers,
            collate_fn=sample_processing_fn, shuffle=self.shuffle_samples
        )
        self.examples_per_sample = parse_nonnegative_int(examples_per_sample,
                                                         'examples_per_sample')
        self.batch_size = parse_nonnegative_int(batch_size, 'batch_size')
        
        # Example sampling and batch collation
        self.shuffle_patches = parse_bool(shuffle_patches, 'shuffle_patches')
        self.drop_last = parse_bool(drop_last, 'drop_last')
        self.example_collate_fn = example_collate_fn
        self.example_queue_maxsize = parse_int(example_queue_maxsize, 
                                               'example_queue_maxsize')
        
        maxsize = self.example_queue_maxsize
        self._example_queue = torch.multiprocessing.Queue(maxsize=maxsize)
        self._example_loader = None
        self._batch_count = None
        self._loading = False
        self._need_queue_clear = False
        self.iter_runs = 0
        
        if headstart:
            self.start_loading()
    
    @property
    def num_samples(self):
        return len(self.dataset)
    
    @property
    def num_batches(self):
        N_examples = self.num_samples * self.examples_per_sample
        N_full_batches = N_examples // self.batch_size
        if self.drop_last:
            return N_full_batches
        mod = N_examples % self.batch_size
        return N_full_batches + bool(mod)
    
    @property
    def examples_per_epoch(self):
        if self.drop_last:
            return self.num_batches * self.batch_size
        return self.num_samples * self.examples_per_sample
    
    @property
    def loading(self):
        return self._loading
    
    @property
    def is_queue_empty(self):
        return self._example_queue.empty()
    
    @property
    def is_queue_full(self):
        return self._example_queue.full()
    
    @property
    def worker_pid(self):
        if self._example_loader is not None:
            return self._example_loader.pid
        return None
    
    @property
    def worker_memory(self):
        if self._example_worker is not None:
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
        self.start_loading()
        return self
    
    def __next__(self):
        if self.batch_count >= self.num_batches:
            self.stop_loading()
            raise StopIteration
        else:
            is_last_batch = self.batch_count == self.num_batches - 1
            if is_last_batch and not self.drop_last:
                batch_size = self.examples_per_epoch % self.batch_size
            else:
                batch_size = self.batch_size
            
            start = time.time()
            examples = []
            for i in range(batch_size):
                example = self._example_queue.get(block=True)
                examples.append(example)
            # print(f'  OTM-Iter: {time.time() - start:.2f}s to get examples.')
            
            start = time.time()
            batch = self.example_collate_fn(examples)  
            # print(f'  OTM-Iter: {time.time() - start:.2f}s to collate batch.')
            self.batch_count += 1
            return batch
    
    
    ### ------ #   Continuous Patch Loading & Multiprocessing  # ----- ###
    
    def start_loading(self):
        not_first_iter = self.iter_runs > 0
        if self.loading and self._example_loader is None:
            raise RuntimeError(f'Loader should be set when loading flag is.')
        
        if self.loading:
            assert self._example_loader is not None
            return
        
        if self._need_queue_clear:
            self._clear_queue(self._example_queue)
            self._need_queue_clear = False
            
        args = (self.sample_loader, self._example_queue, self.shuffle_patches)
        target = example_loading_worker
        self._example_loader = torch_mp.Process(target=target, args=args)
        self._example_loader.start()
        self._loading = True
        logging.info(f'🏗️ OTMLoader: successfully started loading process.')
    
    def stop_loading(self):
        self._loading = False
        if self._example_loader is None:
            return
        if self._example_loader.is_alive():
            self._example_loader.kill()
            self._example_loader.join()
            assert not self._example_loader.is_alive()
            logging.info(f'🏗️ OTMLoader: successfully killed loader process.')
        
        self._example_loader = None
        self.iter_runs += 1
        self._need_queue_clear = True
    
    def _clear_queue(self, Q):
        # clear queue
        from queue import Empty
        try:
            disposal_count = 0
            while True:
                Q.get(block=False)
                disposal_count += 1
                logging.info(f'🏗️ OTMLoader-Main: Clearing item from queue.. '
                             f'Cleared {disposal_count} items.')
        except Exception as e:
            if isinstance(e, Empty):
                logging.info('OTOLoader: Successfully cleared batch queue.')
                logging.info('🏗️ OTMLoader: Successfully cleared batch queue.')
            elif not self.batch_queue.empty():
                raise e
        
        

