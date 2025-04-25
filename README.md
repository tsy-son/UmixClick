## Environment
Training and evaluation environment: Python3.8.8, PyTorch 1.11.0, Ubuntu 20.4, CUDA 11.0. Run the following command to install required packages.
```
pip3 install -r requirements.txt
```
## Demo
<p align="center">
  <img src="./assets/UmixClick Demo.gif" alt="drawing", width="500"/>
</p>


An example script to run the demo. 
```
python3 demo.py --checkpoint=weights/098.pth
```
## Evaluation
Before evaluation, please download the datasets and models, and then configure the path in [config.yml](https://github.com/tsy-son/UmixClick/blob/main/config.yml).

Use the following code to evaluate the huge model.
```
python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=weights/098.pth \
--eval-mode=cvpr \
--datasets=F3
```

## Training
Before training, please download the pretrained weights and configure the dowloaded path in [config.yml](https://github.com/tsy-son/UmixClick/blob/main/config.yml).

The pretrained weight have already been provided：[weight](https://pan.baidu.com/s/16lJZPj_sNCUKYYv5eLs-BA?pwd=b5yh), 请在

Use the following code to train a huge model on C+L: 
```
python train.py models/iter_mask/umixvit_base448_F3_itermask.py
```

## Download 
F3 datasets: [F3](https://pan.baidu.com/s/16lJZPj_sNCUKYYv5eLs-BA?pwd=b5yh)

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

## Acknowledgement
Our project is developed based on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation). Thanks for the nice demo GUI 
