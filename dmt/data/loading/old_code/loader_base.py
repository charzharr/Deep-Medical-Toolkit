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

DEBUG = True


# class SharedCounter(object):
#     """ A synchronized shared counter.

#     The locking done by multiprocessing.Value ensures that only a single
#     process or thread may read or write the in-memory ctypes object. However,
#     in order to do n += 1, Python performs a read followed by a write, so a
#     second process may read the old value before the new one is written by the
#     first process. The solution is to use a multiprocessing.Lock to guarantee
#     the atomicity of the modifications to Value.

#     This class comes almost entirely from Eli Bendersky's blog:
#     http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/

#     """

#     def __init__(self, n = 0):
#         self.count = python_mp.Value('i', n)

#     def increment(self, n = 1):
#         """ Increment the counter by n (default = 1) """
#         with self.count.get_lock():
#             self.count.value += n

#     @property
#     def value(self):
#         """ Return the value of the counter """
#         return self.count.value


# class Queue(mp_queue):
#     """ A portable implementation of multiprocessing.Queue.

#     Because of multithreading / multiprocessing semantics, Queue.qsize() may
#     raise the NotImplementedError exception on Unix platforms like Mac OS X
#     where sem_getvalue() is not implemented. This subclass addresses this
#     problem by using a synchronized shared counter (initialized to zero) and
#     increasing / decreasing its value every time the put() and get() methods
#     are called, respectively. This not only prevents NotImplementedError from
#     being raised, but also allows us to implement a reliable version of both
#     qsize() and empty().

#     """

#     def __init__(self, *args, **kwargs):
#         super(Queue, self).__init__(*args, **kwargs)
#         self.size = SharedCounter(0)

#     def put(self, *args, **kwargs):
#         self.size.increment(1)
#         super(Queue, self).put(*args, **kwargs)

#     def get(self, *args, **kwargs):
#         self.size.increment(-1)
#         return super(Queue, self).get(*args, **kwargs)

#     def qsize(self):
#         """ Reliable implementation of multiprocessing.Queue.qsize() """
#         return self.size.value

#     def empty(self):
#         """ Reliable implementation of multiprocessing.Queue.empty() """
#         return not self.qsize()






class WorkerProcess(python_mp.Process):
    
    def __init__(self, index):
        super().__init__(self)
        self.index = index
        self.pid = os.getpid()
        self.ppid = os.getppid()
        self.exit = python_mp.Event()
        
    def run(self):
        pass
        
    def shutdown(self):
        self.exit.set()


class ExampleWorker:
    # TODO: add self saving attributes after confirm you can use class in Process
    
    def __call__(self, input_queue, example_queue, example_creator_fn):
        
        ppid = os.getppid()
        pid = os.getpid()
        if DEBUG:
            print(f'Child PID {pid} (parent {ppid}) created! ⭐⭐⭐⭐⭐')
            import os, psutil
            process = psutil.Process(os.getpid())
            print(process.memory_info().rss, + ' bytes')  # in bytes 
        
        while True:
            
            try:
                inputs = input_queue.get(block=True)
            except python_mp.Queue.Empty as e:
                print(e)
                print(input_queue)
            
            # Get example from samples
            
            if DEBUG:
                msg = f'Child {pid}: got {len(inputs)} samples.'
                print(msg)
                
            samples = [i for i in inputs]
            example = example_creator_fn(samples)
            example_queue.put(example, block=True)


def worker(input_queue, example_queue, example_creator_fn):
        import os, psutil
        ppid = os.getppid()
        pid = os.getpid()
        if DEBUG:
            print(f'Child PID {pid} (parent {ppid}) created! 👶 👶')
            process = psutil.Process(os.getpid())
            # print(f' Input queue: {input_queue.qsize()}')
            # print(process.memory_info().rss, + ' bytes')  # in bytes 
        
        while True:
            
            inputs = input_queue.get(block=True)
            
            # Get example from samples
            if DEBUG:
                msg = f'Child {pid}: got {len(inputs)} samples.'
                print(msg)
                
            samples = [i for i in inputs]
            start = time.time()
            example = example_creator_fn(samples)
            print(f'👶 Child {pid} - Example created ({time.time() - start:.2f} '
                  f'sec), putting in Q..')
            sys.stdout.flush()
            # example_queue.put(example, block=True)
            example_queue.send(example)



class BatchLoader:
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
        
        # Create sample_set indices 
        self._subject_indices_iterable = None
        self._reset_subject_indices_iterable()
        
        # Multi-processing for example loading (if necessary)
        self.loading = False
        self.num_workers = parse_nonnegative_int(num_workers, 'num_workers')
        self.worker_processes = [None] * num_workers
        self.transfer_daemon, self.daemon_kill = None, None
        
        self.ii = None
        self.queue_manager = None
        self.example_queue_size = parse_nonnegative_int(example_queue_size,
                                                        'example_queue_size')
        self.example_queue = None
        self.batch_loading_queue = None
        self._reset_example_queues()       
        
        self.current_batch_index = 0
        if start_loading:
            self.start_loading()
    
    # @property
    # def loading(self):
    #     num_running = sum([p is not None for p in self.worker_processes])
    #     if num_running == 0:
    #         return False
    #     else:
    #         assert num_running == self.num_workers
    #         return True
    
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
    
    def _get_next_subject_index(self):
        try:
            index = next(self.subject_indices_iterable)
        except StopIteration as exception:
            if DEBUG:
                print(f'Indices list ran dry. Adding more..')
            self._reset_subject_indices_iterable()
            index = next(self.subject_indices_iterable)
        return index
    
    def _reset_subject_indices_iterable(self):
        indices = list(range(len(self.sample_set)))
        if self.shuffle:
            random.shuffle(indices)
        self._subject_indices_iterable = iter(indices)
        
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
    
    def __next__(self):
        """ Grabs the next batch by collating examples in the example-queue.
        Implementation Details
            - Handles drop-last when there's not enough examples for a batch.
            - At end of epoch, kill all processes.
        """
        import itertools
        if self.current_batch_index >= self.num_batches:
            print(f'[Loader] Stopping Iteration')
            import sys; sys.exit(1)
            self.stop_loading()
            raise StopIteration
        else:
            # TODO: check processes are working
            if self.ii is None:
                self.ii = itertools.cycle(range(self.num_workers))
            print(f'[NEW BATCH] !!')
            examples = []
            for i in range(self.examples_per_batch):
                print(f' Getting example {i}..', end=''); start = time.time()
                example = None
                example = self.example_queues[next(self.ii)][1].recv()
                examples.append(example)
                print(f'.done ({time.time() - start} sec)')
            start = time.time()
            batch = self.batch_collate_fn(examples)
            print(f' (__next__) Collate took {time.time() - start:.2f} sec')
            self.current_batch_index += 1
            return batch
    
    # TODO: add function that starts stopped workers (called in __getitem__) 
    # TODO: add line that kills all workers when end of epoch is reached 
    # TODO: add iterator functionality that automatically does the above
    
    def start_loading(self):
        """ Assumes (1) example_queue & batch_load_queue are ready. """
        if self.loading:
            return
        
        worker_indices, worker_samples = self._get_worker_indices_samples()
        self.sample_queues = []
        
        # NEW
        if self.queue_manager is None:
            self.queue_manager = python_mp.Manager()
        self.example_queues = [python_mp.Pipe()] * self.num_workers
        
        for i, process in enumerate(self.worker_processes):
            if process is not None:
                assert False, f'Something is very wrong with worker init.'
            else:
                self.sample_queues.append(python_mp.Queue())
                for ex_samples in worker_samples[i]:
                    self.sample_queues[i].put(ex_samples)
                
                new_process = python_mp.Process(
                    target=worker, 
                    name=f'example_worker_{i}',
                    args=(
                        self.sample_queues[i],
                        self.example_queues[i][0],
                        self.example_creator_fn
                    ))
                self.worker_processes[i] = new_process
                new_process.start()
                
        # start transfer daemon
        # assert self.transfer_daemon is None
        # class Kill: pass
        # self.daemon_kill = Kill()
        # self.transfer_daemon = BatchLoader._start_transfer_daemon(
        #     self.example_queue,
        #     self.batch_loading_queue,
        #     self.daemon_kill
        # )
        self.loading = True
    
    def stop_loading(self):
        # stop loading processes
        for i, process in enumerate(self.worker_processes):
            if process is not None:
                process.terminate()
        for p in self.worker_processes:
            if process is not None:
                p.join()
        self.worker_processes = [None] * self.num_workers
        
        # stop transfer daemon
        # if self.transfer_daemon is not None:
        #     self.daemon_kill = None
        #     self.example_queue.put('kill_me')
        #     assert self.batch_loading_queue is not None
        #     self.transfer_daemon.join()
        
        self.loading = False
        
    def reset_loading(self):
        self.stop_loading()
        self._reset_example_queues()
        self.start_loading()
        
    def _reset_example_queues(self):
        assert not self.loading, 'Can only reset when loading is halted.'
        return
        
        if self.example_queue is not None:
            del self.example_queue
        if self.num_workers >= 1:
            from dmt.utils.queue import FastQueue
            self.example_queue = python_mp.Queue(
                maxsize=self.example_queue_size
            )
        else:
            self.example_queue = queue.Queue(maxsize=self.example_queue_size)
        
        # if self.batch_loading_queue is not None:
        #     del self.batch_loading_queue
        # self.batch_loading_queue = queue.Queue(
        #     maxsize=self.example_queue_size
        # )
            
    @staticmethod
    def _start_transfer_daemon(src_q, dst_q, ref):
        
        def transfer(src_q, dst_q, ref):
            print('😄 Daemon created!')
            while ref():
                start = time.time()
                obj = src_q.get(block=True)
                if isinstance(obj, str) and obj == 'kill_me':
                    break
                print('😄 Putting example into main queue! '
                      f'({time.time() - start:.2f} sec to get)', end='')
                start = time.time()
                dst_q.put(obj)
                print(f' .. ({time.time() - start:.2f} sec to put)')
            print('Daemon is done.')
            
        
        def stop_daemon(ref):
            print(f'(FastQ) Stop thread initiated')
            src_q.put('kill_me')

        wref = weakref.ref(ref, stop_daemon)
        args = (src_q, dst_q, wref,)
        transfer_thread = Thread(target=transfer, args=args)
        transfer_thread.daemon = True
        transfer_thread.start()
        return transfer_thread
    
    
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
    
    
    ### ------ #   Other Functionality   # ----- ###
    
    def __len__(self):
        return self.num_batches - self.current_batch_index
    
    def __repr__(self):
        procs = [False if p is None else True for p in self.worker_processes]
        if len(self.worker_processes) == 0:
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
    
        

        
    

# Tests

def example_creator_fn(samples):
    return [s.image.array for s in samples]

def collate_fn(batch):
    tens_list = [torch.tensor(s[i]) for i in range(len(batch[0])) 
                                    for s in batch]
    return tens_list


if __name__ == '__main__':
    import sys, random
    from pathlib import Path
    import SimpleITK as sitk
    
    curr_path = Path(__file__).absolute()
    base_path = curr_path.parent.parent.parent.parent
    sys.path.append(str(base_path))
    
    from dmt.data import (ScalarImage3D, ScalarMask3D, CategoricalLabel,
                          Sample, SampleSet)
    
    # Get dataset samples
    print('Loading samples..', end=''); sys.stdout.flush(); start = time.time()
    
    train_dir = Path('../../../tests/Task07_Pancreas/imagesTr')
    train_ims = sorted([f for f in os.listdir(train_dir) if f[-2:] =='gz'])
    label_dir = Path('../../../tests/Task07_Pancreas/labelsTr')
    label_ims = sorted([f for f in os.listdir(label_dir) if f[-2:] =='gz'])

    samples = []
    for i in range(len(train_ims))[:34]:
        img_path = train_dir / train_ims[i]
        lab_path = label_dir / label_ims[i]
        name = train_ims[i]
        cns = ['background', 'pancreas', 'tumor']
        image = ScalarImage3D(img_path, sitk.sitkInt16, 
                            permanent_load=False, name=name)
        label = ScalarMask3D(lab_path, cns, container_type=sitk.sitkUInt8)
        label2 = CategoricalLabel(random.randint(0, 2), cns)
        sample = Sample(image=image, label=label, cat=label2, name=name, id=i)
        samples.append(sample)
    
    sampleset = SampleSet(samples)
    print(f'done ({time.time() - start:.2f} sec)')
    
    loader = BatchLoader(
                sampleset,
                samples_per_example=3,
                example_output_size=3,
                examples_per_batch=2, 
                example_creator_fn=example_creator_fn,
                batch_collate_fn=collate_fn, 
                num_workers=4,
                start_loading=True,
                shuffle=False,
                example_queue_size=32,
                )
    print(loader)
    # print(f'\n... Simulating other loading functions..', end='')
    # time.sleep(10); print(f'done ✔')
    import IPython; IPython.embed(); 
    load_start = time.time()
    batch_start = time.time()
    print(f'⭐⭐⭐ Starting loader iteration.')
    for i, batch in enumerate(loader):
        t = time.time() - batch_start
        print(f'\nBatch {i+1} RECEIVED! ({len(batch)} items, {t:.2} sec)')
        for i, t in enumerate(batch):
            print(f'  Batch-Collate Tens {i}, shape: {t.shape}')
        batch_start = time.time()
    print(f'* Final Time: {time.time() - load_start:.2f} sec *')