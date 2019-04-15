# Anonymize Faces for Privacy Preserving
Please, see the following paper/project for details  
[[Project]](https://jason718.github.io/project/privacy/main.html) [[Paper]](https://arxiv.org/abs/1803.11556) 

For Video anonymization training code, please conatact Michael Ryoo ([mryoo@egovid.com](mryoo@egovid.com))

## Results
![results](https://camo.githubusercontent.com/07b708ef957fc86b0ed2bf1b5e2fd667889c551e/68747470733a2f2f6a61736f6e3731382e6769746875622e696f2f70726f6a6563742f707269766163792f66696c65732f7175616c692e6a706567)

## How to use it

### 1. Preparation
First of all, clone the code  
```
git clone https://github.com/blacknwhite5/facial-anonymizer.git
```

Then, create a folder  
```
cd facial-anonymizer && mkdir pretrained
```

### 2. Prerequisites
 * python 3.6
 * pytorch 1.0.0 or higher
 * CUDA 9.0 or higher

### 3. Dependencies
Install all the python Dependencies using pip:  
```
pip install -r requirements.txt
```

> torch    
torchvision  
cython  
cffi  
opencv-python  
scipy  
easydict  
matplotlib  
pyyaml  


### 4. Download pretrained model
Please download bellow models and place models in ```pretrained/``` folder

 * SSH-face detector: [Google Drive](https://drive.google.com/file/d/18AlQ4sqD5hdUOic-zkoC3ldhwCfweOX8/view?usp=sharing)
 * Privacy Preserving : [Google Drive](https://drive.google.com/file/d/1GMouJutcJwlEIxYF_xhG_6ZtDYCrKJwp/view?usp=sharing)

```
 pretrained
    ├── check_point.zip
    └── model_1.pth
```
### 5. Run real-time demo  

You can use a webcam in a real-time demo by running  

```
./scripts/run-demo.sh
```



## License
facial-anonymizer complies with the following license:  
[egovid_license](https://github.com/blacknwhite5/facial-anonymizer/blob/master/egovid_license.txt)