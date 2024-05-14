# SAFINet

# Network Architecture
![image text](https://github.com/WangJianQin7/SAFINet/blob/main/model/SAFINet.png)
# Requirements
python 3.7 + pytorch 1.9.0
# Saliency maps
We provide saliency maps of our SAFINet on [ORSSD](https://pan.baidu.com/s/1ikoNOU2NtuOuP1RLx-Nrzg) (code:SAFI) and [EORSSD](https://pan.baidu.com/s/1tcUdlpTZaZu_evuDiZqROA) (code:SAFI) datasets.
# Training
Run train_SAFINet.py.
# Pre-trained model and testing
Download the following pre-trained model and put them in ./models/SAFINet/, then run test_SAFINet.py.  
[SAFINet_ORSSD](https://pan.baidu.com/s/174HDPJaW86yxvP_Lb8oR1A) (code:SAFI)  
[SAFINet_EORSSD](https://pan.baidu.com/s/1VGgQxyHO0qdm2apvSrZN-g) (code:SAFI)
# Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
