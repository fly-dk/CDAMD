# [Coordinate-Based Dual-Constrained Autoregressive MotionGeneration](https://arxiv.org/abs/2604.08088), [Project](https://fly-dk.github.io/CDAMD/), [arXiv](https://arxiv.org/abs/2604.08088)

[![arXiv](https://img.shields.io/badge/arXiv-<2403.19435>-<COLOR>.svg)](https://arxiv.org/abs/2403.19435)

The official PyTorch implementation of the paper [**"Coordinate-Based Dual-Constrained Autoregressive MotionGeneration"**](https://arxiv.org/abs/2403.19435).

Please visit our [**webpage**](https://fly-dk.github.io/CDAMD/) for more details.


If our project is helpful for your research, please consider citing :
``` 
@inproceedings{xxx,
  title={Coordinate-Based Dual-Constrained Autoregressive MotionGeneration}, 
  author={xxx}, 
  booktitle="xx",
  year={2026},
}
```
## 1. Setup Env & Download Pre-train MoMask
Our model is devloped based on [MoMask](https://github.com/EricGuo5513/momask-codes). You can follow the setup instructions provided there, or refer to the steps below. (The environment setup is the same but we change conda env name to "CDAMD")
<details>
  
### 1. Conda Environment
```
conda env create -f environment.yml
conda activate BAMM
pip install git+https://github.com/openai/CLIP.git
```
We test our code on Python 3.7.13 and PyTorch 1.7.1

#### Alternative: Pip Installation
<details>
We provide an alternative pip installation in case you encounter difficulties setting up the conda environment.

```
pip install -r requirements.txt
```
We test this installation on Python 3.10

</details>

### 2. Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
For evaluation only.
```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

#### Troubleshooting
To address the download error related to gdown: "Cannot retrieve the public link of the file. You may need to change the permission to 'Anyone with the link', or have had many accesses". A potential solution is to run `pip install --upgrade --no-cache-dir gdown`, as suggested on https://github.com/wkentaro/gdown/issues/43. This should help resolve the issue.

#### (Optional) Download Manually
Visit [[Google Drive]](https://drive.google.com/drive/folders/1b3GnAbERH8jAoO5mdWgZhyxHB73n23sK?usp=drive_link) to download the models and evaluators mannually.

### 3. Get Data

You have two options here:
* **Skip getting data**, if you just want to generate motions using *own* descriptions.
* **Get full data**, if you want to *re-train* and *evaluate* the model.

**(a). Full data (text + motion)**

**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the result dataset to our repository:
```
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```
**KIT**-Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then place result in `./dataset/KIT-ML`

#### 

</details>

<!-- ## 2. Download Pre-train BAMM and Move MoMask to "log" folder
```
bash prepare/download_models_BAMM.sh
``` -->

## 2. Train
### 2.1 VQVAE
```
python train_vq.py \
    --dataset_name t2m \
    --name abs_VQVAE_dp1_b256 \
    --batch_size 256 \
    --gpu_id 0
``` 
### 2.2 AE
```
python train_AE.py --name AE --dataset_name t2m --batch_size 256 --epoch 50 --lr_decay 0.05
```
### 2.3 CDAMD
```
python train_abs_transformer.py --dataset_name t2m --name dual_sparse_token_trans_4_all_quants --batch_size 64 --max_epoch 500 --milestones 50000 --trans cross_attn --latent_dim 512 --ff_size 1024 --n_heads 8 --n_layers 4 --gpu_id 0
```
## 4. Evaluation
<details>

```
python eval_t2m_trans_abs.py --name dual_sparse_token_trans_4 --dataset_name t2m --gpu_id 0 --cond_scale 4 --time_steps 10 --ext your_eval --checkpoints_dir ./log/t2m
```
</details>


Please cite our work if you use this code.
```
@misc{ding2026coordinatebaseddualconstrainedautoregressivemotion,
      title={Coordinate-Based Dual-Constrained Autoregressive Motion Generation}, 
      author={Kang Ding and Hongsong Wang and Jie Gui and Liang Wang},
      year={2026},
      eprint={2604.08088},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.08088}, 
}
```
