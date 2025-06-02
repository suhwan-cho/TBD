# TBD

This is the official PyTorch implementation of our paper:

> **Tackling Background Distraction in Video Object Segmentation**, *ECCV 2022*\
> Suhwan Cho, Heansung Lee, Minhyeok Lee, Chaewon Park, Sungjun Jang, Minjung Kim, Sangyoun Lee\
> Link: [[ECCV]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820434.pdf) [[arXiv]](https://arxiv.org/abs/2207.06953)

<img src="https://github.com/user-attachments/assets/ab20ec8e-c984-4e2f-84c2-b7d5a389ba1e" width=800>

You can also explore other related works at [awesome-video-object segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In semi-supervised VOS, one of the main challenges is the existence of background distractors that have a similar appearance to the target objects. As 
comparing visual properties is a fundamental technique, visual distractions can severely lower the reliability of a system. To suppress the negative 
influence of background distractions, we propose three novel strategies: 1) a **spatio-temporally diversified template construction scheme** to prepare 
various object properties for reliable and stable prediction; 2) a **learnable distance-scoring function** to consider the temporal consistency of a 
video; 3) **swap-and-attach data augmentation** to provide hard training samples showing severe occlusions.


## Setup
1\. Download the datasets:
[COCO](https://cocodataset.org/#download),
[DAVIS](https://davischallenge.org/davis2017/code.html),
[YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data).

2\. Download our [custom split](https://drive.google.com/drive/folders/14FcZXKjqIVoO375w3_bH6YcQI9GuKYIf?usp=drive_link) for the YouTube-VOS training set.




## Running 

### Training
Start TBD training with:
```
python run.py --train
```

Verify the following before running:\
✅ Training dataset selection and configuration\
✅ GPU availability and configuration


### Testing
Run TBD with:
```
python run.py --test
```

Verify the following before running:\
✅ Testing dataset selection\
✅ GPU availability and configuration\
✅ Pre-trained model path


## Attachments
[Pre-trained model (davis)](https://drive.google.com/file/d/1bL5sew3b76XhXe7vlC57gYZSPjkbmaPb/view?usp=drive_link)\
[Pre-trained model (ytvos)](https://drive.google.com/file/d/1mDHAj5Utih9aNmmexsXzoUGZQzzeb-Tf/view?usp=drive_link)\
[Pre-computed results](https://drive.google.com/file/d/1d-KIWYBxbU-VHmLqvCbyLMz_sRExt8Ir/view?usp=drive_link)


## Contact
Code and models are only available for non-commercial research purposes.\
For questions or inquiries, feel free to contact:
```
E-mail: suhwanx@gmail.com
```
