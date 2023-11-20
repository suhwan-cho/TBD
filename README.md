# TBD

This is the official PyTorch implementation of our paper:

> **Tackling Background Distraction in Video Object Segmentation**, *ECCV 2022*\
> Suhwan Cho, Heansung Lee, Minhyeok Lee, Chaewon Park, Sungjun Jang, Minjung Kim, Sangyoun Lee\
> Link: [[ECCV]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820434.pdf) [[arXiv]](https://arxiv.org/pdf/2207.06953.pdf)

<img src="https://user-images.githubusercontent.com/54178929/208470682-c36bfd92-db65-47ce-8e7e-7da51937686f.png" width=800>

You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In semi-supervised VOS, one of the main challenges is the existence of background distractors that have a similar appearance to the target objects. As 
comparing visual properties is a fundamental technique, visual distractions can severely lower the reliability of
a system. To suppress the negative influence of background distractions, we propose three novel strategies: 1) a **spatio-temporally diversified template
construction scheme** to prepare various object properties for reliable and stable prediction; 2) a **learnable distance-scoring function** to consider the temporal
consistency of a video; 3) **swap-and-attach data augmentation** to provide hard training samples showing severe occlusions.


## Preparation
1\. Download [COCO](https://cocodataset.org/#download) for network pre-training.

2\. Download [DAVIS](https://davischallenge.org/davis2017/code.html) and [YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data) for network main training and testing.

3\. Download [Custom Split](https://drive.google.com/drive/folders/1R5Z0aQQw2lvsoAlqtHLjY4RYNXg7SJkX?usp=drive_link) for YouTube-VOS training samples for network validation.

4\. Replace dataset paths in *"run.py"* file with your dataset paths.


## Training
1\. Move to "run.py" file.

2\. Check the training settings.

3\. Run **TBD** training!!
```
python run.py --train
```

## Testing
1\. Move to "run.py" file.

2\. Select a pre-trained model.

3\. Run **TBD** testing!!
```
python run.py --test
```


## Attachments
[pre-trained model (davis)](https://drive.google.com/file/d/1KFCd1jc1752hI1yf1OEMchR3ND292Xyh/view?usp=sharing)\
[pre-trained model (ytvos)](https://drive.google.com/file/d/107H-VB-QgCY9mHTgiw2E5_j9IA9QApIN/view?usp=sharing)\
[pre-computed results](https://drive.google.com/file/d/1tHe7YqYSo_F8W3ISP6pjh4RTG_bOg4sb/view?usp=drive_link)


## Note
Code and models are only available for non-commercial research purposes.\
If you have any questions, please feel free to contact me :)
```
E-mail: chosuhwan@yonsei.ac.kr
```
