""" Module dmt/data/loading/loader_base.py

Loading Concepts:
    Given:
     - SampleSet & custom/default preprocessing
     - 
     
Assumptions:
    - # of samples per example & # of tensors per example is same for epoch

samples = [Sample(file) for file in os.listdir('dataset')]
sample_set = dmt.data.SampleSet(samples)


"""

import os, sys
import pathlib
import time
import random
import queue
import weakref
import warnings
import multiprocessing as python_mp
import threading
from threading import Thread
from numpy import BUFSIZE

import torch

if __name__ == '__main__':
    python_mp.set_start_method('spawn')  # spawn
    curr_path = pathlib.Path(__file__)
    sys.path.append(str(curr_path.parent.parent.parent.parent))

from dmt.utils.parse import (
    parse_nonnegative_int, parse_positive_int,
    parse_bool
)
from dmt.data.loading.loader_worker import Worker


DEBUG = True



class BatchLoader2:
    """ Base class for serialized or multi-processing data loading. 
    """
    
    def __init__(
            self,
            sample_set,
            example_creator_fn,
            samples_per_example,
            example_output_size,
            batch_collate_fn, 
            examples_per_batch, 
            num_workers=4,
            example_queue_size=16,
            start_loading=False,
            shuffle=False
            ):
        """
        Args:
            sample_set: collection of data samples to iterate over.
            example_creator_fn: function to create examples from samples
                See sample & example abstractions in README. 
            begin_loading: flag to start loading examples right after init. 
                Usually, this only starts when __getitem__ is called. 
            shuffle: flag to shuffle samples. 
        """
        
        self.sample_set = sample_set
        self.shuffle = parse_bool(shuffle, 'shuffle')
        
        self.example_creator_fn = example_creator_fn
        self.samples_per_example = parse_nonnegative_int(
                                        samples_per_example, 
                                        'samples_per_example')
        
        # Batch creation from examples
        self.batch_collate_fn = batch_collate_fn
        self.examples_per_batch = parse_nonnegative_int(
                                        examples_per_batch, 
                                        'examples_per_batch')
        self.example_output_size = parse_positive_int(
                                        example_output_size,
                                        'example_output_size')
        
        # Multi-processing for example loading (if necessary)
        self.loading = False
        self.num_workers = parse_nonnegative_int(num_workers, 'num_workers')
        self.workers = [None] * self.num_workers
        
        self.example_queue_size = parse_nonnegative_int(example_queue_size,
                                                        'example_queue_size')
        self.example_queues = [None] * 2
        self.feed_queues = [None] * self.num_workers
        self._reset_example_queues()       
        
        self.current_batch_index = 0
        if start_loading:
            self.start_loading()
    
    
    @property
    def num_subjects(self):
        return len(self.sample_set)
    
    @property
    def num_examples(self):
        return self.num_subjects // self.samples_per_example
    
    @property
    def num_batches(self):
        num_batches = self.num_examples // self.examples_per_batch
        return num_batches
    
    @property
    def batch_size(self):
        return self.examples_per_batch * self.example_output_size
    
    
    ### ------ #   Sample Iteration Functionality   # ----- ###
        
    def _get_worker_indices_samples(self):
        indices = list(range(len(self.sample_set)))
        if self.shuffle:
            random.shuffle(indices)
        
        spe = self.samples_per_example
        examples = [indices[spe*i:spe*i+spe] for i in range(len(indices)//spe)]
        nw = self.num_workers
        worker_indices = [examples[i::nw] for i in range(nw)]
        worker_samples = []
        for wi, worker_inds in enumerate(worker_indices):
            worker_samples.append([])
            for ex_inds in worker_inds:
                worker_samples[wi].append([self.sample_set[i] for i in ex_inds])
        
        return worker_indices, worker_samples
    
    
    ### ------ #   Example Creation Functionality   # ----- ###
    
    # TODO: add function that starts stopped workers (called in __getitem__) 
    # TODO: add line that kills all workers when end of epoch is reached 
    # TODO: add iterator functionality that automatically does the above
    
    def start_loading(self):
        """ Assumes (1) example_queue & batch_load_queue are ready. """
        if self.loading:
            return
        for q in self.feed_queues:
            assert q is not None
        for q in self.example_queues:
            assert q is not None
        
        worker_indices, worker_samples = self._get_worker_indices_samples()
        self.sample_queues = []
        for i, worker in enumerate(self.workers):
            if worker is not None:
                assert False, f'Something is very wrong with worker init.'
            else:
                example_queue = self.example_queues[i % 2]
                feed_queue = self.feed_queues[i]
                assert feed_queue.empty()
                
                for ex_samples in worker_samples[i]:
                    feed_queue.put(ex_samples)
                
                new_worker = Worker(i, example_queue, self.example_creator_fn,
                                    feed_queue, start_loading=True)
                self.workers[i] = new_worker
        self.loading = True
    
    def stop_loading(self):
        # stop loading processes
        for i, worker in enumerate(self.workers):
            if worker is not None:
                worker.kill()
        self.workers = [None] * self.num_workers
        self.loading = False
        
    def reset_loading(self):
        self.stop_loading()
        self._reset_example_queues()
        self.start_loading()
        
    def _reset_example_queues(self):
        assert not self.loading, 'Can only reset when loading is halted.'
        
        # create queues if doesn't exist
        for i, q in enumerate(self.example_queues):
            if q is None:
                self.example_queues[i] = queue.Queue(
                    maxsize=self.example_queue_size)
        for i, q in enumerate(self.feed_queues):
            if q is None:
                self.feed_queues[i] = python_mp.Queue(
                    maxsize=self.example_queue_size)
        
        # empty the queues
        for i, q in enumerate(self.example_queues):
            self._clear_queue(q)
        for i, q in enumerate(self.feed_queues):
            self._clear_queue(q)
        
    def _clear_queue(self, queue):
        if queue is None:
            return
        from queue import Empty
        try:
            while True:
                queue.get_nowait()
        except Empty:
            return
    
    
    ### ------ #   Core API & Batch Iteration   # ----- ###
    
    def __iter__(self):
        """ Initialize signal from user code. 
            0. Checks if example loading has already started or not.
            1. If not started, load sample queues for each worker.
                Also checks if all workers are functioning properly.
            2. Initialize workers to start filling example-queue.
        """
        if not self.loading:
            self.reset_loading()
        self.current_batch_index = 0
        return self
    
    def __next__(self):
        """ Grabs the next batch by collating examples in the example-queue.
        Implementation Details
            - Handles drop-last when there's not enough examples for a batch.
            - At end of epoch, kill all processes.
        """
        
        if self.current_batch_index >= self.num_batches:
            print(f'[Loader] Stopping Iteration')
            self.stop_loading()
            raise StopIteration
        else:
            # TODO: check processes are working
            print(f'[NEW BATCH] !!')
            examples = []
            qn = 0
            for i in range(self.examples_per_batch):
                print(f' (__next__) Getting example {i+1}..', end=''); start = time.time()
                example = self.example_queues[qn].get()
                qn = 0 if qn == 1 else 1
                examples.append(example)
                print(f' (__next__) Got example {i+1} ({time.time() - start} sec)')
            batch = self.batch_collate_fn(examples)
            self.current_batch_index += 1
            return batch
    
    
    ### ------ #   Other Functionality   # ----- ###
    
    def __len__(self):
        return self.num_batches - self.current_batch_index
    
    def __repr__(self):
        procs = [False if p is None else True for p in self.workers]
        if len(self.workers) == 0:
            procs = False
        string = (
            f'BatchLoader Object (workers={self.num_workers}) \n'
            f'  Workers Running: {str(procs)} \n'
            f'  SampleSet size = {len(self.sample_set)}, '
              f'Samples/Ex = {self.samples_per_example}, '
              f'Ex/Batch = {self.examples_per_batch} \n'
            f'  {self.num_batches} Batches w/ Size {self.batch_size}, '
              f'= Ex/Batch {self.examples_per_batch} * Tensors/Ex '
              f'{self.example_output_size}'
        )
        return string