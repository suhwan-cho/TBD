# TBD

This is the official PyTorch implementation of our paper:

> **Tackling Background Distraction in Video Object Segmentation**, *ECCV 2022*\
> Suhwan Cho, Heansung Lee, Minhyeok Lee, Chaewon Park, Sungjun Jang, Minjung Kim, Sangyoun Lee\
> Link: [[ECCV]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820434.pdf) [[arXiv]](https://arxiv.org/pdf/2207.06953.pdf)

<img src="https://github.com/user-attachments/assets/ab20ec8e-c984-4e2f-84c2-b7d5a389ba1e" width=800>

You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In semi-supervised VOS, one of the main challenges is the existence of background distractors that have a similar appearance to the target objects. As 
comparing visual properties is a fundamental technique, visual distractions can severely lower the reliability of a system. To suppress the negative 
influence of background distractions, we propose three novel strategies: 1) a **spatio-temporally diversified template construction scheme** to prepare 
various object properties for reliable and stable prediction; 2) a **learnable distance-scoring function** to consider the temporal consistency of a video; 
3) **swap-and-attach data augmentation** to provide hard training samples showing severe occlusions.


## Preparation
1\. Download 
[COCO](https://cocodataset.org/#download), 
[DAVIS](https://davischallenge.org/davis2017/code.html),
and [YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data)
from the official websites.

2\. Download our [custom split](https://drive.google.com/drive/folders/14FcZXKjqIVoO375w3_bH6YcQI9GuKYIf?usp=drive_link) for the YouTube-VOS training set.

3\. Replace dataset paths in "run.py" file with your dataset paths.


## Training
1\. Open the "run.py" file.

2\. Verify the training settings.

3\. Start **TBD** training!
```
python run.py --train
```

## Testing
1\. Open the "run.py" file.

2\. Choose a pre-trained model.

3\. Start **TBD** testing!
```
python run.py --test
```


## Attachments
[pre-trained model (davis)](https://drive.google.com/file/d/1bL5sew3b76XhXe7vlC57gYZSPjkbmaPb/view?usp=drive_link)\
[pre-trained model (ytvos)](https://drive.google.com/file/d/1mDHAj5Utih9aNmmexsXzoUGZQzzeb-Tf/view?usp=drive_link)\
[pre-computed results](https://drive.google.com/file/d/1d-KIWYBxbU-VHmLqvCbyLMz_sRExt8Ir/view?usp=drive_link)


## Note
Code and models are only available for non-commercial research purposes.\
If you have any questions, please feel free to contact me :)
```
E-mail: suhwanx@gmail.com
```
