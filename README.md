# mindspore-cleanrl
MindSpore version of [CleanRL](https://github.com/vwxyzjn/cleanrl)(775), for supporting online reinforcement learning algorithms

## :wrench: Dependencies

- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- MindSpore == 1.9.0

### Installation

1. **Clone repo**

   ```bash
   git clone https://github.com/superboySB/mindspore-cleanrl.git && cd mindspore-cleanrl
   ```

2. [Optional] Create Virtual Environment for GPU

   ```sh
   sudo apt-get install libgmp-dev
   wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
   sudo sh ./cuda_11.1.1_455.32.00_linux.run --override
   echo -e "export PATH=/usr/local/cuda-11.1/bin:\$PATH" >> ~/.bashrc
   echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
   source ~/.bashrc
   
   # cudnn needs license
   wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-linux-x64-v8.0.5.39.tgz
   tar -zxvf cudnn-11.1-linux-x64-v8.0.5.39.tgz
   sudo cp cuda/include/cudnn.h /usr/local/cuda-11.1/include
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64
   sudo chmod a+r /usr/local/cuda-11.1/include/cudnn.h /usr/local/cuda-11.1/lib64/libcudnn*
   
   # Install mindspore-gpu
   conda create -n msrl python==3.8
   conda activate msrl
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.9.0-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **Install minimal dependent packages**

   ```shell
   # Install mindspore-cpu (Not all machines have GPU)
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/cpu/x86_64/mindspore-1.9.0-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # Install full dependendies for cleanrl
   pip install -r requirements_ms.txt
   
   # if you want to compare the same files in mindspore/torch/jax, run this to install them:
   pip intall -r requirements_others.txt
   ```

4. [Optional] All unit tests can be run using `pytest` runner:

   ```
   make pytest
   ```


## :computer: Example

We can compare the performance of the same file in different ML packages, by just changing directories: mindspore -> torch/jax.

```shell
# classic control
python cleanrl/mindspore/dqn.py --env-id CartPole-v1
python cleanrl/mindspore/ppo.py --env-id CartPole-v1
```


## :checkered_flag: Testing & Rendering

We will evaluate the trained model here.

```
\comming soon!
```

## :page_facing_up: Q&A

Q: Cannot render the results

> libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)

A: Modify the conda env

```sh
cd /home/$USER/miniconda3/envs/msrl/lib
mkdir backup  # Create a new folder to keep the original libstdc++
mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.29
```

## :clap: Reference

This codebase is based on CleanRL and msrl which are open-sourced. Please refer to that repo for more documentation.

- CleanRL (https://github.com/vwxyzjn/cleanrl)
- MindSpore/reinforcement (msrl) (https://gitee.com/mindspore/reinforcement)

## :e-mail: Contact

If you have any question, please email `daizipeng@bit.edu.cn`.
