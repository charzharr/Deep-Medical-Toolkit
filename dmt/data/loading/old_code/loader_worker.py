
import os, sys
import time
import weakref
import threading
import multiprocessing
from queue import Empty


DEBUG = True


def process_fn(input_queue, pipe_send, example_creator_fn):
    ppid = os.getppid()
    pid = os.getpid()
    if DEBUG:
        print(f'Child PID {pid} (parent {ppid}) created! 👶 👶')
        # process = psutil.Process(os.getpid())
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
                f'sec), putting in pipe..')
        start = time.time()
        sys.stdout.flush()
        # pipe_send.send(example)
        pipe_send.put(example)
        print(f'  Child {pid} - Sent (pipe send took {time.time() - start:.2f} sec.')


def thread_fn(pipe_receive, example_queue, kill_ref):
    """ Transfers piped completed-examples to an example_queue which
        is a thread-safe queue that's called in the main thread & loader. """
    print(f'😄 Thread Daemon created!!')
    
    while True:
        try:
            # example = pipe_receive.recv()
            example = pipe_receive.get(block=False)
        except Empty:
            time.sleep(0.1)
            continue
        except EOFError():
            print(f'Pipe closed triggered thread death')
            break
        print(f'😄 Thread Daemon received object!!!')
        
        # if not kill_ref():
        #     print(f'Kill reference has triggered thread death.')
        #     break
        if isinstance(example, str) and example == 'kill_thread':
            print(f'Piped signal has signaled thread death.')
            break
        print(f'Putting shit in queue')
        start = time.time()
        example_queue.put(example, block=True)
        print(f'Put that shit in queue ({time.time() - start:.2f} sec).')
    print(f'😄 Thread Daemon exited!!')


class Worker:
    
    def __init__(
            self,
            index,
            example_output_queue, 
            example_creator_fn,
            process_input_queue, 
            start_loading=False
            ):
        self.index = index
        self.example_output_queue = example_output_queue
        self.example_creator_fn = example_creator_fn
        self.process_input_queue = process_input_queue
        # self.pipe_rec, self.pipe_snd = multiprocessing.Pipe(duplex=False)
        self.pipe_rec = self.pipe_snd = multiprocessing.Queue()
        
        self.loading_process = None
        self.transfer_thread = None
        self.thread_kill = None
        if start_loading:
            self.start()
            
    @property
    def is_thread_running(self):
        if self.transfer_thread is None or not self.transfer_thread.is_alive():
            return False
        return True
    
    @property
    def is_process_running(self):
        if self.loading_process is None or not self.loading_process.is_alive():
            return False
        return True
        
    def start(self):
        self._start_loading_process(self.pipe_snd, self.process_input_queue)
        self._start_transfer_daemon(self.pipe_rec, self.example_output_queue)
    
    def kill(self):
        self._kill_transfer_daemon()
        self._kill_loading_process()
        
    def _start_loading_process(self, pipe_send, input_queue):
        if self.is_process_running:
            return
        args = (input_queue, pipe_send, self.example_creator_fn)
        self.loading_process = multiprocessing.Process(
            target=process_fn, 
            name=f'worker_process_{self.index}',
            args=args
        )
        self.loading_process.start()
    
    def _start_transfer_daemon(self, pipe_receive, output_queue):
        if self.is_thread_running:
            return
        class KillPill: pass
        wref = weakref.ref(KillPill())
        self.thread_kill = wref
        
        args = (pipe_receive, output_queue, wref,)
        self.transfer_thread = threading.Thread(target=thread_fn, args=args)
        self.transfer_thread.daemon = False
        self.transfer_thread.start()
        
    def _kill_transfer_daemon(self):
        if not self.is_thread_running:
            return
        self.thread_kill = None
        self.pipe_snd.send('kill_thread')
        self.pipe_snd.close()
        self.transfer_thread.join()
        self.transfer_thread = None
    
    def _kill_loading_process(self):
        if not self.is_process_running:
            return
        self.loading_process.terminate()
        self.loading_process.join()
        self.loading_process = None
    
    def _clear_queue(self, queue):
        if queue is None:
            return
        from queue import Empty
        try:
            while True:
                queue.get_nowait()
        except Empty:
            return
        
    def __del__(self):
        self._kill_transfer_daemon()
        self._kill_loading_process()
        self.example_output_queue = None
        self.process_input_queue = None
        self.pipe_rec, self.pipe_snd = None, None




