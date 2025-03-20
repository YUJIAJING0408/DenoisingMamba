# Denoising Mamba

****

## Dependencies (other versions may also work)

```shell
pip install -r requirements.txt
```

## Dataset

![dataset](https://github.com/YUJIAJING0408/DenoisingMamba/tree/master/images/dataset.jpg)

The KJL dataset is provided by ACFM and AFGSA, which contains a total of 1109 [Tungsten](https://github.com/tunabrain/tungsten) shots with the noisy images rendered at 32spp and the gt images rendered at 32768spp.These file can be found from [BaiduYunPan](https://github.com/tunabrain/tungsten).

The YBC dataset is rendered by us. The YBC dataset provides noise maps of 1, 2, 4, 8, 16, 32, 64, 128 SPP, as well as clear maps of 16K and corresponding depth normal reflections. We will upload the dataset to Baidu Netdisk soon

## Model

![network_1](https://github.com/YUJIAJING0408/DenoisingMamba/tree/master/images/network_1.jpg)

The architecture of Denoising Mamba is shown in the above figure. Firstly, a fast Fourier convolution extractor (FFCE) is used to extract auxiliary features and noise features from the auxiliary buffer and noise image, respectively. To further enhance the capability of long-distance modeling, a residual multi-path architecture Mamba encoder (RMME) is designed, which combines auxiliary features and noise features to input into RMME. The internal architecture of RMME is shown in the following figure, which fully utilizes horizontal and vertical information through a cross four-way scanning scheme.

![network_2](https://github.com/YUJIAJING0408/DenoisingMamba/tree/master/images/network_2.jpg)



## Model weights

The base-model have 3 RMME layers with 3,4,5 division-size. We train our base-model for 1500 epochs by 120*120 image. It can be found from [BaiduYunPan](https://github.com/tunabrain/tungsten).

## Train and Inference

##### Train

```shell
python train.py --img_size 120 --dataset "dataset_path" --data_name "ybc" --epochs 400 --model "DM" --log_path "log_path"
```

##### Inference

```shell
python inferences/dm-ybc.py -td "test_dataset" -is 120 -o "output_path" -m "model_path" -d cuda
```

# Result

![models.png](https://github.com/YUJIAJING0408/DenoisingMamba/tree/master/images/models.png)

Inference Cost

![mem-time.png](https://github.com/YUJIAJING0408/DenoisingMamba/tree/master/images/mem-time.png)
