
import sys, pathlib
import torch
import time

curr_path = pathlib.Path(__file__).absolute()
sys.path.append(str(curr_path.parent.parent.parent))
# from dmt.data.loading.old_code.loader_base import BatchLoader
# from dmt.data.loading.old_code.loader_v0 import BatchLoader0
# from dmt.data.loading.old_code.loader_v2 import BatchLoader2
# from dmt.data.loading.old_code.loader_v3 import BatchLoader3
    

def example_creator_fn(samples):
    return [{'image': s.image.array, 'label': s.label.array} for s in samples]

def collate_fn(batch):
    batch_d = {k: [] for k in batch[0][0].keys()}
    for ex in batch:
        for d in ex:
            for k, v in d.items():
                batch_d[k].append(torch.tensor(v))
    return batch_d

def load(sample_set, num_workers=4):
    loader = BatchLoader3(
                sample_set,
                samples_per_example=3,
                example_output_size=3,
                examples_per_batch=2, 
                example_creator_fn=example_creator_fn,
                batch_collate_fn=collate_fn, 
                num_workers=num_workers,
                start_loading=True,
                shuffle=False,
                example_queue_size=32
                )
    print(repr(loader) + '\n')
    time.sleep(5)
    
    batch_start = time.time()
    print(f' Starting loader iteration.')
    for i, batch in enumerate(loader):
        t = time.time() - batch_start
        print(f'\n  Batch {i+1} RECEIVED! ({len(batch)} items, {t:.2f} sec)')
        for k in batch.keys():
            print(f'  {k}: ', end='')
            for t in batch[k]:
                print(f'{t.shape}', end='')
        print()
        batch_start = time.time()