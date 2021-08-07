"""
Main module to run in order to time data loading of large files (esp 3D imgs). 
This module is in charge of:
    - organizing the data to be loaded into samples & a sampleset 
    - calling loading methods within the same directory
    - timing and comparing multiple runs (disregarding 1st run bc of caching)
Compares the loading methods: 
    - completely serialized (pytorch defualt loader w/ 0 workers)
    - pytorch multi-processing (4 workers)
    - dmt-loading (1 worker)
    - dmt-loading (4 workers)

Time Recordings (100 images, batch_size=6, img + masks) 7/23
    - torch-loader-4: Average time 74.27 sec (17.15 sec for fork)
    - dmt-loader-2: 
    
    - daemon-loader-4: fork 13.89 sec 
    - torch-loader-4 (fork): spawn 76.57
"""

import os, sys
import psutil
import random
import time
import pathlib
import SimpleITK as sitk
import numpy as np
import multiprocessing as python_mp

curr_path = pathlib.Path(__file__).absolute()
if __name__ == '__main__':
    python_mp.set_start_method('fork')  # spawn
    sys.path.append(str(curr_path.parent.parent.parent))  # project main dir
    
from dmt.data import (ScalarImage3D, ScalarMask3D, CategoricalLabel,
                      Sample, SampleSet)
from dmt_loading import load as dmt_load
from torch_loading import load as torch_load

# Experiment Variables
msd_dataset_path = pathlib.Path('../Task07_Pancreas')
preload_data = False
dataset_size = 100
run_n_times = 2




if __name__ == '__main__':
    
    # Get dataset samples
    print('Loading samples..', end=''); sys.stdout.flush(); start = time.time()
    
    train_dir = msd_dataset_path / 'imagesTr'
    train_ims = sorted([f for f in os.listdir(train_dir) if f[-2:] =='gz'])
    label_dir = msd_dataset_path / 'labelsTr'
    label_ims = sorted([f for f in os.listdir(label_dir) if f[-2:] =='gz'])

    samples = []
    for i in range(len(train_ims))[:dataset_size]:
        img_path = train_dir / train_ims[i]
        lab_path = label_dir / label_ims[i]
        name = train_ims[i]
        cns = ['background', 'pancreas', 'tumor']
        image = ScalarImage3D(img_path, sitk.sitkInt16, 
                            permanent_load=preload_data, name=name)
        label = ScalarMask3D(lab_path, cns, container_type=sitk.sitkUInt8)
        label2 = CategoricalLabel(random.randint(0, 2), cns)
        sample = Sample(image=image, label=label, cat=label2, name=name, id=i)
        samples.append(sample)
    
    sampleset = SampleSet(samples)
    print(f'done ({time.time() - start:.2f} sec)\n')
    pid = os.getpid()
    this_process = psutil.Process(pid)
    mem_usage = this_process.memory_info()[0]/2.**30
    print(f'Using {mem_usage:.2f} GB in PID {pid}')
    
    print()
    print('Testing Torch Loader (4 workers)'.center(70, '-'))
    
    dmt4_times = np.zeros(run_n_times)
    for i in range(run_n_times):
        start = time.time()
        torch_load(sampleset, num_workers=4)
        elapsed = time.time() - start
        print(f' ⭐ Finished run {i+1} in {elapsed:.2f} sec')
        dmt4_times[i] = elapsed
    print(f' ⭐ ⭐ Average time {dmt4_times.mean():.2f} sec')
    
    import sys; sys.exit(1)
    print()
    print('Testing DMT-Loader (4 Workers)'.center(70, '-'))
    
    dmt4_times = np.zeros(run_n_times)
    for i in range(run_n_times):
        start = time.time()
        dmt_load(sampleset, num_workers=4)
        elapsed = time.time() - start
        print(f' ⭐ Finished run {i+1} in {elapsed:.2f} sec')
        dmt4_times[i] = elapsed
    print(f' ⭐ ⭐ Average time {dmt4_times.mean():.2f} sec')