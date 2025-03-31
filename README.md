# Denoising Mamba

****

<center>Jiajing Yu, Rui Zhu,Tongzhou Zhao </center>

## Abstract

Monte Carlo (MC) denoising plays a pivotal role in enhancing image quality in physically-based rendering. Despite the widespread adoption of learning-based methods, achieving high denoising accuracy while minimizing GPU memory usage remains a formidable challenge. In this paper, we introduce Denoising Mamba, a novel approach that leverages a memory-efficient state space model (SSM) architecture, circumventing the limitations of attention-based methods. Our framework incorporates an optimized Fast Fourier Feature extractor for seamless local-global feature fusion and a Residual Multiway Mamba Encoder (RMME) that captures long-range spatial dependencies through linear-complexity bidirectional scanning. This culminates in a lightweight decoder for reconstructing high-resolution, denoised images. Experimental results on real datasets demonstrate a notable reduction in $RMSE^{-3}$ (by 2.554), an increase in $PSNR$ (by 2.04 dB), and an improvement in $SSIM^{-2}$ (by 5.04), while requiring only 15% of the GPU memory and 40% of the inference time compared to state-of-the-art attention-based methods. Our code and datasets are publicly available at [[GitHub - YUJIAJING0408/DenoisingMamba: Denoising Mamba](https://github.com/YUJIAJING0408/DenoisingMamba)].

## Dependencies

```shell
pip install -r requirements.txt
```

### Envs

```
python==3.10.3
torch==2.1.1
lightning==2.1.1
cuda==2.1
tensorboard>= 2.18.0
```

## Dataset

![dataset](https://github.com/YUJIAJING0408/DenoisingMamba/blob/master/images/dataset.jpg)

The KJL dataset is provided by [ACFM](https://github.com/mcdenoising/AdvMCDenoise) and [AFGSA](https://github.com/Aatr0x13/MC-Denoising-via-Auxiliary-Feature-Guided-Self-Attention), which contains a total of 1109 [Tungsten](https://github.com/tunabrain/tungsten) shots with the noisy images rendered at 32spp and the gt images rendered at 32768spp.These file can be found from [BaiduYunPan](https://pan.baidu.com/s/1Jyck8eOcdc7aI-P3gvEnBw?pwd=YBCD).

The YBC dataset is rendered by us. The YBC dataset provides noise maps of 1, 2, 4, 8, 16, 32, 64, 128 SPP, as well as clear maps of 16K and corresponding depth normal reflections. [BaiduYunPan](https://pan.baidu.com/s/1Jyck8eOcdc7aI-P3gvEnBw?pwd=YBCD).

## Model

![network_1](https://github.com/YUJIAJING0408/DenoisingMamba/blob/master/images/network_1.jpg)

The architecture of Denoising Mamba is shown in the above figure. Firstly, a fast Fourier convolution extractor (FFCE) is used to extract auxiliary features and noise features from the auxiliary buffer and noise image, respectively. To further enhance the capability of long-distance modeling, a residual multi-path architecture Mamba encoder (RMME) is designed, which combines auxiliary features and noise features to input into RMME. The internal architecture of RMME is shown in the following figure, which fully utilizes horizontal and vertical information through a cross four-way scanning scheme.

![network_2](https://github.com/YUJIAJING0408/DenoisingMamba/blob/master/images/network_2.jpg)

## Model weights

The base-model have 3 RMME layers with 3,4,5 division-size. We train our base-model for 1500 epochs by 120*120 image. It can be found from [BaiduYunPan](https://github.com/tunabrain/tungsten).

### Train

```shell
python train.py --img_size 120 --dataset "dataset_path" --data_name "ybc" --epochs 400 --model "DM" --log_path "log_path"
```

### Monitoring

```shell
tensorboard --logdir "output_dir" --load_fast true
```

Turn To [TensorBoard]( http://localhost:6006/)

### Inference

```shell
python inferences/dm-ybc.py -td "test_dataset" -is 120 -o "output_path" -m "model_path" -d cuda
```

## Result

![models.png](https://github.com/YUJIAJING0408/DenoisingMamba/blob/master/images/models.png)

Inference Cost

![mem-time.png](https://github.com/YUJIAJING0408/DenoisingMamba/blob/master/images/mem-time.png)

## Citation

```
@proceedings{denoising_mamba,
  author = {Rui,Zhu and Jiajing, Yu and Tongzhou, Zhao},
  title = {Efficient Monte Carlo Denoising via State Space Model with Low GPU Memory Overhead},
  journal = {The Visual Compute},
  year = {2025}
}
```
