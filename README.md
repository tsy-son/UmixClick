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
Before training, please download the pretrained weights,

Use the following code to train a huge model on C+L: 
```
python train.py models/iter_mask/umixvit_base448_F3_itermask.py
```

## Download 
F3 datasets: [F3](https://pan.baidu.com/s/16lJZPj_sNCUKYYv5eLs-BA?pwd=b5yh)

Pretrained weight：[weight](https://pan.baidu.com/s/1rntFPCclSi5OMst_1hFBtQ?pwd=ucv1), This weight needs to be added at the end of [umit.py]([UmixClick/isegm/model/modeling/umit.py at main · tsy-son/UmixClick](https://github.com/tsy-son/UmixClick/blob/main/isegm/model/modeling/umit.py)
