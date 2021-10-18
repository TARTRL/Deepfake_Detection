# Deepfake Detection

### Introduction

- Use deep models to detect fake images. 

- Distributed training with Pytorch for deepfake detection.

- Check out this [repo](https://github.com/EndlessSora/DeeperForensics-1.0) to know more about deepfake images.

- Real image: 

![](./docs/images/0.png)

- Fake image:

![](./docs/images/1.jpeg)

### Install
- Python 3+
- `pip install -r requirements.txt`
- `pip install -e .`

### Training
```
cd scripts
./train.sh
```

### Testing
You need download the pretrained model from BaiduYun:

Download link: https://pan.baidu.com/s/16NIV5BVUITKwolQzPbj9Zw  Password:`r3i8`

And put the model(`model_half.pth.tar`) 
under `./models`. Then you can run python script as below:
```
cd scripts
./test.sh
```

### Cite
Please cite our code if you use this code or our models in your own work:
```
@misc{deepfake_detection,
  title={Deepfake Detection},
  author={Huang, Shiyu},
  howpublished={\url{https://github.com/tartrl/deepfake_detection}},
  year={2020}
}
```