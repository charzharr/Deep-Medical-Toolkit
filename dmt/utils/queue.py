""" Module dmt/utils/queue.py

Implements custom multi-processing queue that is faster for big data transfer
and avoids the problem of slow buffer speeds in Q.get() in multiprocessing.
"""

import multiprocessing
import queue
import weakref
from threading import Thread


class FastQueue:
    """ Wraps mp.Queue.get() with a thread daemon that moves it immediately
        to a threaded queue so no buffering is necessary when get is called.
    """
    
    def __init__(self, maxsize=0):
        
        if maxsize <= 0:
            import _multiprocessing
            maxsize = _multiprocessing.SemLock.SEM_VALUE_MAX
        self.maxsize = maxsize
        
        self.mpq = multiprocessing.Queue(maxsize=maxsize)
        self.qq = queue.Queue(maxsize=maxsize)
        
        FastQueue._start_transfer_daemon(self.mpq, self.qq, self)

    def __del__(self):
        del self.mpq
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
        