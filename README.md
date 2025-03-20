# Denoising Mamba

****

## Dependencies (other versions may also work)

```shell
pip install -r requirements.txt
```

## Dataset

The KJL dataset is provided by ACFM and AFGSA, which contains a total of 1109 [Tungsten](https://github.com/tunabrain/tungsten) shots with the noisy images rendered at 32spp and the gt images rendered at 32768spp.These file can be found from [BaiduYunPan](https://github.com/tunabrain/tungsten).

The YBC dataset is rendered by us. The YBC dataset provides noise maps of 1, 2, 4, 8, 16, 32, 64, 128 SPP, as well as clear maps of 16K and corresponding depth normal reflections. We will upload the dataset to Baidu Netdisk soon

## Model





## Model weights

The base-model have 3 RMME layers with 3,4,5 division-size. We train our base-model for 1500 epochs by 120*120 image. It can be found from [BaiduYunPan](https://github.com/tunabrain/tungsten).

## Train and Inference

##### Train

```shell
python train.py --img_size 120 --dataset "dataset_path" --data_name "ybc" --epochs 400 --model "DM" --log_path "log_path"
```

##### Inference

```shell
python inferences/mffmc.py -td "test_dataset" -is 120 -o "output_path" -m "model_path"-d cuda
```
