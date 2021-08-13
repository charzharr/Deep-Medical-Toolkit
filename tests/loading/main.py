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

Time Recordings (100 images, batch_size=6, img + masks, 1sec sleep iter) 7/23
    - torch-loader-4: Average time 74.27 sec (17.15 sec for fork)
    - oto-loader-4: 60.5 sec (13.89 sec for fork)
    
    (100 images, 3 crops each, batch_size=16, no sleep iter) 8/13
    - tio-loader-4: 63 sec for fork
    - 
"""

import os, sys
import psutil
import random
import time
import torchio
import pathlib
import multiprocessing
import SimpleITK as sitk
import numpy as np
import multiprocessing as python_mp

curr_path = pathlib.Path(__file__).absolute()
if __name__ == '__main__':
    python_mp.set_start_method('fork')  # spawn
    sys.path.append(str(curr_path.parent.parent.parent))  # project main dir
    
from dmt.data import (ScalarImage3D, ScalarMask3D, CategoricalLabel,
                      Sample, SampleSet)
from torch_loading import load as torch_load
from oto_loading import load as oto_load
from otm_loading import load as otm_load
from tio_loading import load as tio_load

# Experiment Variables
msd_dataset_path = pathlib.Path('../Task07_Pancreas')
preload_data = False
dataset_size = 100
run_n_times = 2

test_torch_loader = False
test_oto_loader = False
test_tio_loader = False
test_otm_loader = True

def get_sample(args):
    i, name, cns, img_path, lab_path = args
    image = ScalarImage3D(img_path, sitk.sitkInt16, 
                            permanent_load=preload_data, name=name)
    label = ScalarMask3D(lab_path, cns, container_type=sitk.sitkUInt8)
    label2 = CategoricalLabel(random.randint(0, 2), cns)
    sample = Sample(image=image, label=label, cat=label2, name=name, id=i)
    
    subject = torchio.Subject(
        image = torchio.ScalarImage(img_path),
        mask = torchio.LabelMap(lab_path),
        id=i,
    )
    
    return sample, subject

if __name__ == '__main__':
    
    # Get dataset samples
    print('Loading samples..', end=''); sys.stdout.flush(); start = time.time()
    
    train_dir = msd_dataset_path / 'imagesTr'
    train_ims = sorted([f for f in os.listdir(train_dir) if f[-2:] =='gz'])
    label_dir = msd_dataset_path / 'labelsTr'
    label_ims = sorted([f for f in os.listdir(label_dir) if f[-2:] =='gz'])
    
    with multiprocessing.Pool() as pool:
        args = []
        for i in range(len(train_ims))[:dataset_size]:
            img_path = train_dir / train_ims[i]
            lab_path = label_dir / label_ims[i]
            name = train_ims[i]
            cns = ['background', 'pancreas', 'tumor']
            args.append((i, name, cns, img_path, lab_path))
        samples = pool.map(get_sample, args)
    
    sampleset = SampleSet([s[0] for s in samples])
    subjectset = torchio.SubjectsDataset([s[1] for s in samples],
                    transform=torchio.Compose([
                        torchio.RandomFlip(),
                        torchio.RandomBlur(),
                        torchio.RandomNoise(),
                    ]))
    print(f'done ({time.time() - start:.2f} sec)\n')
    pid = os.getpid()
    this_process = psutil.Process(pid)
    mem_usage = this_process.memory_info()[0]/2.**30
    print(f'Using {mem_usage:.2f} GB in PID {pid}')
    
    if test_torch_loader:
        print()
        print('Testing Torch Loader (4 workers)'.center(70, '-'))
        
        dmt4_times = np.zeros(run_n_times)
        for i in range(run_n_times):
            start = time.time()
            torch_load(sampleset, num_workers=4)
            elapsed = time.time() - start
            print(f'\n ⭐ Finished run {i+1} in {elapsed:.2f} sec\n')
            dmt4_times[i] = elapsed
        print(f'\n ⭐ ⭐ Average time {dmt4_times.mean():.2f} sec\n')
    
    if test_oto_loader:
        print()
        print('Testing OTO-Loader (4 Workers)'.center(70, '-'))
        
        dmt4_times = np.zeros(run_n_times)
        for i in range(run_n_times):
            start = time.time()
            oto_load(sampleset, num_workers=4)
            elapsed = time.time() - start
            print(f'\n ⭐ Finished run {i+1} in {elapsed:.2f} sec\n')
            dmt4_times[i] = elapsed
        print(f'\n ⭐ ⭐ Average time {dmt4_times.mean():.2f} sec\n')

    if test_tio_loader:
        print()
        print('Testing OTM-Loader (4 Workers)'.center(70, '-'))
        
        tio4_times = np.zeros(run_n_times)
        for i in range(run_n_times):
            start = time.time()
            tio_load(subjectset, num_workers=4)
            elapsed = time.time() - start
            print(f'\n ⭐ Finished run {i+1} in {elapsed:.2f} sec\n')
            tio4_times[i] = elapsed
        print(f'\n ⭐ ⭐ Average time {tio4_times.mean():.2f} sec\n')
    
    if test_otm_loader:
        print()
        print('Testing OTM-Loader (4 Workers)'.center(70, '-'))
        
        otm4_times = np.zeros(run_n_times)
        for i in range(run_n_times):
            start = time.time()
            otm_load(sampleset, num_workers=4)
            elapsed = time.time() - start
            print(f'\n ⭐ Finished run {i+1} in {elapsed:.2f} sec\n')
            otm4_times[i] = elapsed
        print(f'\n ⭐ ⭐ Average time {otm4_times.mean():.2f} sec\n')
        
   