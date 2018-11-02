# Computer Vision 관련 논문 읽기 


---

## 완료된 튜토리얼: 

1. Xception - 설명 & Pytorch 코드
    - ipynb 가 로딩이 안되는 경우: [여기에 있습니다](https://nbviewer.jupyter.org/github/Hyunjulie/KR-Reading-Image-Segmentation-Papers/blob/master/Xception%EC%84%A4%EB%AA%85%EA%B3%BC%20Pytorch%EA%B5%AC%ED%98%84.ipynb) 
2. GoTurn - 설명 & Pytorch 코드
    - ipynb 가 로딩이 안되는 경우: [여기에 있습니다](https://nbviewer.jupyter.org/github/Hyunjulie/KR-Reading-Image-Segmentation-Papers/blob/master/GoTurn%EC%84%A4%EB%AA%85_Pytorch.ipynb)
3. 구글 Colab 에서 Pytorch로 Pretrained 된 모델 사용하는 방법 등 
    - 이것 저것 유용할 수 있는 command 필기용 
    - ipynb 가 로딩이 안되는 경우: [여기에 있습니다](https://nbviewer.jupyter.org/github/Hyunjulie/KR-Reading-Image-Segmentation-Papers/blob/master/Pytorch_Using_Pretrained.ipynb)
4. SegNet - Pytorch 코드 



------

### 목표: 
* Backbone Network 몇가지 정리: Xception, Inception 
* Image Segmentation - 논문 정리하기
* Reuse 가능한 function들 만들고 (PyTorch) 
* 논문 구현하기
+) 최근에 나온 새로운 논문 읽고 정리하기 

--- 

### 논문/모델 목록: TO DO!!

Pre-requisites: ResNet, Inception

- [FCN](https://arxiv.org/pdf/1411.4038.pdf) (2014)
- [SegNet](https://arxiv.org/abs/1511.00561) (2015)
- [MobileNet](https://arxiv.org/abs/1704.04861) (2017)
- [PSPNet](https://arxiv.org/abs/1612.01105) (2016): pyramid scene parsing network 
- [The One Hundred Layers Tiramisu](https://arxiv.org/abs/1611.09326): Fully Convolutional DenseNets for Semantic Segmentation (2016): 
- [DeepLab V3+](https://arxiv.org/abs/1802.02611) (2018)
- [RefineNet](https://arxiv.org/abs/1611.06612) (2016): 
- [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323) (2016)
- [Global Convolutional Network](https://arxiv.org/abs/1703.02719) (2017)
- [AdapNet](http://ais.informatik.uni-freiburg.de/publications/papers/valada17icra.pdf) (2017): 
- [ICNet](https://arxiv.org/abs/1704.08545) (2017): 
- [SenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html) (2018 CVPR): 
- [Dense Decoder Shortcut Connections for Single-Pass Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/html/Bilinski_Dense_Decoder_Shortcut_CVPR_2018_paper.html) (2018 CVPR): 
- [BiSeNet](https://arxiv.org/abs/1808.00897) (2018): 



[참고]
- https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
