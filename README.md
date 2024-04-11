# A UAV visible-Infrared Dual-modality Image Alignment Benchmark


## Install the environment
**Option1**: Use the Anaconda
```
conda create -n UAVmatch python=3.7
conda activate UAVmatch
bash install.sh

# Install Deformable Attention CUDA
cd lib/models/stark_dual_deform/ops
sh ./make.sh

# unit test (should see all checking is True)
python test.py
```



## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${ROOT}
    -- data
        -- DroneVehicle
            |-- rtest          
            |-- rtest_hom        
            |-- test
            |-- train
            |-- val
        -- VTUAV
            |-- test_LT
            |-- train
            |-- LT_train_split.txt
            |-- init_frame.npy
        -- VEDAI
            |-- Vehicules512
            |-- Annotatiobs512

   ```
## Set project paths

you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train UAVmatch
Training with multiple GPUs using DDP
```
python tracking/train.py --script stark_dual_deformer --config baseline --save_dir /workspace --mode multiple --nproc_per_node 4
python tracking/train.py --script stark_dual_deformer --config baseline_hom --save_dir /workspace --mode multiple --nproc_per_node 4
```
(Optionally) Debugging training with a single GPU
```
python lib/train/run_training.py  --script stark_dual_deformer --config baseline
python lib/train/run_training.py  --script stark_dual_deformer --config baseline_hom
```
## Download Weight and Dataset
```
[Baidu Netdisk](https://pan.baidu.com/s/1WAYKxeJQDp_IeCW28qfpPA)  code：gfkd
Weight   :  UAVmatch/weight.zip
Dataset  :  UAVmatch/rtest.zip or rtest_hom.zip
toolkit  :  UAVmatch/toolkit.zip
```

## weight
```
Affine transformer :  weight/stark_dual_deform/match.pth.tar
Hom transformer    :  weight/stark_dual_deform_hom/match.pth.tar
```

copy to:
```
lib/test/parameter/stark_dual_deformer.py
```

## Fast test (NO Need Dataset)
- DroneVehicle

```
lib/train/admin/local.py  # Set self.Dtest_dir = '/you path.../UAVmatch/pic/DroneVehicle'
python tracking/test.py stark_dual_deformer baseline --dataset Dtest  --threads 0 
```


## Test and evaluate UAVmatch on benchmarks
- DroneVehicle_pubilsh   [Baidu Netdisk](https://pan.baidu.com/s/1WAYKxeJQDp_IeCW28qfpPA)  code：gfkd
```
python tracking/test.py stark_dual_deformer baseline --dataset DroneVehicle_norandom --threads 4 
```

- DroneVehicle   [DroneVehicle](https://github.com/VisDrone/DroneVehicle)
```
python tracking/test.py stark_dual_deformer baseline --dataset DroneVehicle --threads 4 
```

- VTUAV          [VTUAV](https://zhang-pengyu.github.io/DUT-VTUAV/)        
```
python tracking/test.py stark_dual_deformer baseline --dataset VTUAV --threads 4 
```
- VEDAI         [VEDAI](https://downloads.greyc.fr/vedai/)
```
python tracking/test.py stark_dual_deformer baseline --dataset VEDAI --threads 0 
```

