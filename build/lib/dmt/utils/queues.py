""" Module dmt/utils/queue.py

Implements custom multi-processing queue that is faster for big data transfer
and avoids the problem of slow buffer speeds in Q.get() in multiprocessing.
"""

import logging
import random
import multiprocessing
import torch
import queue
import weakref
import threading
from threading import Thread


class Counter:
    """ Process-safe counter. """
    def __init__(self, start_value=0):
        self._count = multiprocessing.Value('i', 0)
        
    @property
    def count(self):
        return self._count.value
        
    def increment(self, n=1):
        with self._count.get_lock():
            self._count.value += n
    
    def decrement(self, n=1):
        with self._count.get_lock():
            self._count.value -= n


class RandomQueue:
    """ A process-safe Queue that returns a random element within
        the queue. This is done via random priorities give in put().
    Components:
        - torch.multiprocessing.Queue
        - queue.PriorityQueue
        - daemon that uses a separate thread to load from Q to PQ
    """
    
    def __init__(self, maxsize=0, shuffle=True):
        import _multiprocessing
        python_maxsize = _multiprocessing.SemLock.SEM_VALUE_MAX
        if maxsize <= 0:
            maxsize = python_maxsize
        self._maxsize = min(maxsize, python_maxsize // 2)
        
        self._queue = torch.multiprocessing.Queue(maxsize=self._maxsize)
        self._pqueue = torch.multiprocessing.Queue(maxsize=self._maxsize)
        # self._pqueue = queue.PriorityQueue(maxsize=self._maxsize)
        self._size = Counter(start_value=0)
        
        self.kill_event = threading.Event()
        self.shuffle = shuffle
        self.loading = False
        self.transfer_thread = None
        self.start_loading()
    
    def put(self, item, block=True, timeout=None):
        self._queue.put(item, block=block, timeout=timeout)
        self._size.increment()
        
    def get(self, block=True, timeout=None):
        item = self._pqueue.get(block=block, timeout=timeout)
        self._size.decrement()
        return item[1]
    
    def get_nowait(self):
        item = self._pqueue.get_nowait()
        self._size.decrement()
        return item[1]

    def qsize(self):
        return self._size.count
    
    def empty(self):
        return True if self._size.count == 0 else False

    def full(self):
        return True if self._size.count == self._maxsize else False
    
    def start_loading(self):
        if self.loading:
            return
        args = self._queue, self._pqueue, self, self.kill_event
        self.transfer_thread = RandomQueue._start_transfer_daemon(*args)
        self.loading = True
        
    def delete(self):
        self._queue.put('kill_me')
        clear_queue(self._pqueue)
        while self.transfer_thread.is_alive():
            time.sleep(0.25)
        assert not self.transfer_thread.is_alive()
        self.transfer_thread = None
    
    @staticmethod
    def _start_transfer_daemon(src_q, dst_q, me, kill_event):
        
        def transfer(src_q, dst_q, kill_event):
            wait_flag = True
            transfer_items = set()
            while not kill_event.is_set():
                try:
                    obj = src_q.get(block=False, timeout=0.1)
                except:
                    obj = None
                # if isinstance(obj, str) and obj == 'kill_me':
                #     print(f'(FastQ-daemon) Got Kill. Quitting daemon thread.')
                #     break
                if obj is not None:
                    transfer_items.add(obj)
                    
                if not transfer_items or (wait_flag and len(transfer_items) < 24):
                    continue
                else:
                    wait_flag = False
                    priority = random.randint(1, 1000000)
                    item = transfer_items.pop()
                    priority_item = (priority, item)
                    dst_q.put(priority_item)
            print(f'Transfer thread daemon killed.')
            logging.info('RandomQueue\'s transfer daemon process is exiting')
        
        def stop_daemon(ref):
            print(f'(FastQ) Stop thread initiated')
            # src_q.put('kill_me')
            kill_event.set()
        
        me_weak = weakref.ref(me, stop_daemon)
        args = (src_q, dst_q, kill_event)
        transfer_thread = Thread(target=transfer, args=args)
        transfer_thread.daemon = True
        transfer_thread.start()
        return transfer_thread
    
    
        
    
        

"""
class FastQueue:
    # Wraps mp.Queue.get() with a thread daemon that moves it immediately
    # to a threaded queue so no buffering is necessary when get is called.
    
    def __init__(self, maxsize=0):
        if maxsize <= 0:
            import _multiprocessing
            maxsize = _multiprocessing.SemLock.SEM_VALUE_MAX
        self.maxsize = maxsize
        
        self.mpq = torch.multiprocessing.Queue(maxsize=maxsize)
        self.qq = queue.Queue(maxsize=maxsize)

        FastQueue._start_transfer_daemon(self.mpq, self.qq, self)

    def __del__(self):
        self.mpg.close()
        del self.mpq
        self.qq.close()
        del self.qq

    def put(self, item, block=True, timeout=None):
        self.mpq.put(item, block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        return self.qq.get(block=block, timeout=timeout)

    def qsize(self):
        return self.qq.qsize() + self.mpq.qsize()

    def empty(self):
        return self.qq.empty() and self.mpq.empty()

    def full(self):
        return self.qq.full() and self.mpq.full()

    @staticmethod
    def _start_transfer_daemon(src_q, dst_q, me):
        sentinel = object()

        def transfer(src_q, dst_q, me_ref):
            while me_ref():
                obj = src_q.get(block=True)
                if obj is sentinel:
                    print(f'(FastQ-daemon) End reached. Quitting daemon thread.')
                    break
                dst_q.put(obj)
                    
                # print 'steal'
            # print 'daemon done'
        
        def stop_daemon(ref):
            print(f'(FastQ) Stop thread initiated')
            src_q.put(sentinel)

        me1 = weakref.ref(me, stop_daemon)
        transfer_thread = Thread(target=transfer, args=(src_q, dst_q, me1,))
        transfer_thread.daemon = True
        transfer_thread.start()
"""

def clear_queue(queue):
    from queue import Empty
    try:
        while True:
            queue.get_nowait()
    except Empty:
        return


if __name__ == '__main__':
    print('hi')
    rq = RandomQueue()
    print(f'RandomQueue created (maxsize = {rq._maxsize})')
    
    import time
    start = time.time()
    rq.put(torch.rand((128, 128, 64)))
    rq.put(torch.rand((128, 128, 65)))
    rq.put(torch.rand((128, 128, 66)))
    rq.put(torch.rand((128, 128, 67)))
    rq.put(torch.rand((128, 128, 68)))
    rq.put(torch.rand((512, 512, 69)))
    rq.put(torch.rand((512, 512, 70)))
    time.sleep(2)
    
    while not rq.empty():
        print(rq.get().shape)
    print(f'Took {time.time() - start:2f} sec')
    rq.delete()
        