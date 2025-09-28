# GloMER: Towards Robust Multimodal Emotion Recognition via Gated Fusion and Contrastive Learning
<i>
  Official code repository for the manuscript 
  <b>"GloMER: Towards Robust Multimodal Emotion Recognition via Gated Fusion and Contrastive Learning"</b>, 
  accepted to 
  <a href="https://conferences.sigappfr.org/medes2025/">The 17th International Conference on Management of Digital Ecosystems</a>.
</i>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/nhut-ngnn/GloMER">
<img src="https://img.shields.io/github/forks/nhut-ngnn/GloMER">
<img src="https://img.shields.io/github/watchers/nhut-ngnn/GloMER">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.8.20-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-28.09.2025-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>


<div align="center">

[**Abstract**](#Abstract) •
[**Install**](#install) •
[**Usage**](#usage) •
[**References**](#references) •
[**Citation**](#citation) •
[**Contact**](#Contact)

</div>

## Abstract 
> Speech Emotion Recognition (SER) enhances Human-Computer Interaction (HCI) across healthcare, education, and customer service. Although multimodal approaches that combine audio and text show promise, they face challenges in aligning heterogeneous modalities and achieving balanced feature fusion. In this work, we propose GloMER, a novel multimodal SER architecture that integrates a self-alignment strategy with contrastive learning and a Gated Multimodal Unit (GMU) for adaptive fusion. The self-alignment mechanism ensures semantic consistency while preserving modality diversity. The GMU dynamically regulates modality contributions, ensuring balanced integration and context-aware representations under imbalanced conditions. We evaluate GloMER, achieving state-of-the-art (SOTA) on two benchmark datasets, delivering consistent improvements in multimodal emotion recognition. Ablation studies show that combining classification with alignment improves embedding discriminability, while gated fusion balances modalities and captures complementary cues. These findings establish GloMER as a robust framework for advancing multimodal emotion recognition.
>
> Index Terms: Gated Multimodal Unit, Contrastive Learning, Human-Computer Interaction, Multimodal Emotion Recognition.


## Install
### Clone this repository
```
git clone https://github.com/nhut-ngnn/GloMER.git
```

### Create Conda Enviroment and Install Requirement
Navigate to the project directory and create a Conda environment:
```
cd GloMER
conda create --name GloMER python=3.8
conda activate GloMER
```
### Install Dependencies
```
pip install -r requirements.txt
```
### Dataset 

GloMER is evaluated on two widely-used multimodal emotion recognition datasets:

#### IEMOCAP (Interactive Emotional Dyadic Motion Capture)
- **Modality**: Audio + Text  
- **Classes**: `angry`, `happy`, `sad`, `neutral` (4 classes)  
- **Sessions**: 5  
- **Official Website**: [https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)  
- **Note**: We use **Wav2Vec2.0** for audio and **BERT** for text feature extraction.

#### ESD (Emotional Speech Dataset)
- **Modality**: Audio + Text  
- **Languages**: English, Mandarin, and more  
- **Classes**: `neutral`, `angry`, `happy`, `sad`, `surprise` (5 classes)  
- **Official GitHub**: [https://github.com/HLTSingapore/ESD](https://github.com/HLTSingapore/ESD)  

## Citation
If you use this code or part of it, please cite the following papers:
```
Nhut Minh Nguyen, Duc Tai Phan, and Duc Ngoc Minh Dang, “GloMER: Towards Robust Multimodal Emotion Recognition via Gated Fusion and Contrastive Learning”, The 17th International Conference on Management of Digital EcoSystems (MEDES 2025), Ho Chi Minh City, Vietnam, Nov 24-26, 2025.
```

## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

**Email:** [minhnhut.ngnn@gmail.com](mailto:minhnhut.ngnn@gmail.com)<br>
**ORCID:** <link>https://orcid.org/0009-0003-1281-5346</link> <br>
**GitHub:** <link>https://github.com/nhut-ngnn/</link>