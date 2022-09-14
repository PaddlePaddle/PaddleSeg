<center>
	<img src="https://user-images.githubusercontent.com/50255927/189930324-0f3992cd-47f8-487c-b20e-5a59f28f978f.png" width="30%" />
	&emsp;&emsp;&emsp;
	<img src="https://user-images.githubusercontent.com/50255927/189930368-64d62295-559e-4982-a228-40873156b8a0.png" width="30%" />
    &emsp;&emsp;&emsp;
    <img src="https://user-images.githubusercontent.com/50255927/189930342-d32b90e5-ef80-44fb-9eab-c9df25ca0d12.png" width="30%" />
</center>

***

# MUSCLE - MICCAI 2022
This is a repository for paper "MUSCLE: Multi-task Self-supervised Continual Learning to 
Pre-train Deep Models for X-ray Images of Multiple Body Parts" accepted by MICCAI 2022.

## Introduction
***
The goal of MUSCLE (*<u>MU</u>lti-task <u>S</u>elf-supervised <u>C</u>ontinual <u>LE</u>arning*) is to pre-train the deep neural network (DNN) models and 
deliver decent performance on medical image analysis tasks.

+ All codes are implemented using PaddlePaddle.

## Framework
***
![image](https://user-images.githubusercontent.com/50255927/189317770-c8c9e866-beb2-4eb5-8116-21ab00850ef0.png)

MUSCLE aggregated multiple X-ray image datasets collected from different human body
parts, subject to various X-ray analytics tasks. We proposed Multi-Dataset Momentum 
Contrastive Representation Learning (MD-MoCo) and Multi-task Continual Learning to 
pre-train the backbone DNNs in a self-supervised continual learning manner. 
The pre-trained models could be fine-tuned to target tasks using task-specific heads 
and achieve superb performance.

## Datasets
***
<table class="tg">
<thead>
  <tr>
    <th class="tg-cly1">Datasets</th>
    <th class="tg-cly1">Body Part</th>
    <th class="tg-cly1">Task</th>
    <th class="tg-cly1">Train</th>
    <th class="tg-cly1">Valid</th>
    <th class="tg-cly1">Test</th>
    <th class="tg-cly1">Total</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" colspan="7">Only Used for the first step (MD-MoCo) of MUSCLE</td>
  </tr>
  <tr>
    <td class="tg-cly1">NIHCC</td>
    <td class="tg-nrix">Chest</td>
    <td class="tg-nrix">N/A</td>
    <td class="tg-cly1">112,120</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-cly1">112,120</td>
  </tr>
  <tr>
    <td class="tg-cly1">China-Set-CXR</td>
    <td class="tg-nrix">Chest</td>
    <td class="tg-nrix">N/A</td>
    <td class="tg-cly1">661</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-cly1">661</td>
  </tr>
  <tr>
    <td class="tg-cly1">Montgomery-Set-CXR</td>
    <td class="tg-nrix">Chest</td>
    <td class="tg-nrix">N/A</td>
    <td class="tg-cly1">138</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-cly1">138</td>
  </tr>
  <tr>
    <td class="tg-cly1">Indiana-CXR</td>
    <td class="tg-nrix">Chest</td>
    <td class="tg-nrix">N/A</td>
    <td class="tg-cly1">7,470</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-cly1">7,470</td>
  </tr>
  <tr>
    <td class="tg-cly1">RSNA Bone Age</td>
    <td class="tg-nrix">Hand</td>
    <td class="tg-nrix">N/A</td>
    <td class="tg-cly1">10,811</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-mwxe">N/A</td>
    <td class="tg-cly1">10,811</td>
  </tr>
  <tr>
    <td class="tg-nrix" colspan="7">Used for all three steps of MUSCLE</td>
  </tr>
  <tr>
    <td class="tg-cly1">Pneumonia</td>
    <td class="tg-nrix">Chest</td>
    <td class="tg-nrix">Classification</td>
    <td class="tg-cly1">4,686</td>
    <td class="tg-cly1">585</td>
    <td class="tg-cly1">585</td>
    <td class="tg-cly1">5,856</td>
  </tr>
  <tr>
    <td class="tg-cly1">MURA</td>
    <td class="tg-nrix">Various Bone</td>
    <td class="tg-nrix">Classification</td>
    <td class="tg-cly1">32,013</td>
    <td class="tg-cly1">3,997</td>
    <td class="tg-cly1">3,997</td>
    <td class="tg-cly1">40,005</td>
  </tr>
  <tr>
    <td class="tg-cly1">Chest Xray Masks and labels</td>
    <td class="tg-nrix">Chest</td>
    <td class="tg-nrix">Segmentation</td>
    <td class="tg-cly1">718</td>
    <td class="tg-cly1">89</td>
    <td class="tg-cly1">89</td>
    <td class="tg-cly1">896</td>
  </tr>
  <tr>
    <td class="tg-cly1">TBX</td>
    <td class="tg-nrix">Chest</td>
    <td class="tg-nrix">Detection</td>
    <td class="tg-cly1">640</td>
    <td class="tg-cly1">80</td>
    <td class="tg-cly1">80</td>
    <td class="tg-cly1">800</td>
  </tr>
  <tr>
    <td class="tg-cly1">Total</td>
    <td class="tg-nrix">N/A</td>
    <td class="tg-nrix">N/A</td>
    <td class="tg-cly1">169,257</td>
    <td class="tg-cly1">4,751</td>
    <td class="tg-cly1">4,479</td>
    <td class="tg-cly1">178,757</td>
  </tr>
</tbody>
</table>

## Experiments
***
### Experiment setups
- Backbone
    - ResNet-18 and ResNet-50
- Task
    - Pneumonia classification (Pneumonia), 
    - Skeletal abnormality classification (MURA)
    - Lung segmentation (Lung)
    - Tuberculosis detection (TBX)
- Head
    - Fully-Connected (FC) Layer for classification tasks
    - DeepLab-V3 for segmentation tasks
    - FasterRCNN for detection tasks
- Baselines Pre-training Algorithms
    - **Scratch**: the models are all initialized using Kaiming’s random initialization and fine-tuned on the target datasets
    - **ImageNet**: the models are initialized using the officially-released weights pre-trained by the ImageNet dataset and fine-tuned on the target datasets
    - **MD-MoCo**: the models are pre-trained using multi-dataset MoCo and fine-tuned accordingly
    - **MUSCLE−−**: all models are pre-trained and fine-tuned with MUSCLE but with Cross-Task Memorization and Cyclic and Reshuffled Learning Schedule turned off

### Results for various body parts
+ Note: **chest** of Pneumonia and **bones** of MURA

<div>

<table class="tg">
<thead>
  <tr>
    <th class="tg-8d8j">Datasets</th>
    <th class="tg-2b7s">Backbones</th>
    <th class="tg-7zrl">Pre-train</th>
    <th class="tg-2b7s">Acc.</th>
    <th class="tg-8d8j">Sen.</th>
    <th class="tg-8d8j">Spe.</th>
    <th class="tg-2b7s">AUC(95%CI)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8d8j" rowspan="10">Pneumonia</td>
    <td class="tg-2b7s" rowspan="5">ResNet-18</td>
    <td class="tg-7zrl">Scratch</td>
    <td class="tg-2b7s">91.11</td>
    <td class="tg-8d8j">93.91</td>
    <td class="tg-8d8j">83.54</td>
    <td class="tg-2b7s">96.58(95.09-97.81)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ImageNet</td>
    <td class="tg-2b7s">90.09</td>
    <td class="tg-8d8j">93.68</td>
    <td class="tg-8d8j">80.38</td>
    <td class="tg-2b7s">96.05(94.24-97.33)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MD-MoCo</td>
    <td class="tg-2b7s">96.58</td>
    <td class="tg-8d8j">97.19</td>
    <td class="tg-8d8j">94.94</td>
    <td class="tg-2b7s">98.48(97.14-99.30)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE--</td>
    <td class="tg-2b7s">96.75</td>
    <td class="tg-8d8j">97.66</td>
    <td class="tg-8d8j">94.30</td>
    <td class="tg-2b7s">99.51(99.16-99.77)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE</td>
    <td class="tg-2b7s">97.26</td>
    <td class="tg-8d8j">97.42</td>
    <td class="tg-8d8j">96.84</td>
    <td class="tg-2b7s">99.61(99.32-99.83)	</td>
  </tr>
  <tr>
    <td class="tg-2b7s" rowspan="5">ResNet-50</td>
    <td class="tg-7zrl">Scratch</td>
    <td class="tg-2b7s">91.45</td>
    <td class="tg-8d8j">92.51</td>
    <td class="tg-8d8j">88.61</td>
    <td class="tg-2b7s">96.55(95.08-97.82)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ImageNet</td>
    <td class="tg-2b7s">95.38</td>
    <td class="tg-8d8j">95.78</td>
    <td class="tg-8d8j">94.30</td>
    <td class="tg-2b7s">98.72(98.03-99.33)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MD-MoCo</td>
    <td class="tg-2b7s">97.09</td>
    <td class="tg-8d8j">98.83</td>
    <td class="tg-8d8j">92.41</td>
    <td class="tg-2b7s">99.53(99.23-99.75)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE--</td>
    <td class="tg-2b7s">96.75</td>
    <td class="tg-8d8j">98.36</td>
    <td class="tg-8d8j">92.41</td>
    <td class="tg-2b7s">99.58(99.30-99.84)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE</td>
    <td class="tg-2b7s">98.12</td>
    <td class="tg-8d8j">98.36</td>
    <td class="tg-8d8j">97.47</td>
    <td class="tg-2b7s">99.72(99.46-99.92)</td>
  </tr>
  <tr>
    <td class="tg-8d8j" rowspan="10">MURA</td>
    <td class="tg-2b7s" rowspan="5">ResNet-18</td>
    <td class="tg-7zrl">Scratch</td>
    <td class="tg-2b7s">81.00</td>
    <td class="tg-8d8j">68.17</td>
    <td class="tg-8d8j">89.91</td>
    <td class="tg-2b7s">86.62(85.73-87.55)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ImageNet</td>
    <td class="tg-2b7s">81.88</td>
    <td class="tg-8d8j">73.49</td>
    <td class="tg-8d8j">87.70</td>
    <td class="tg-2b7s">88.11(87.18-89.03)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MD-MoCo</td>
    <td class="tg-2b7s">82.48</td>
    <td class="tg-8d8j">72.27</td>
    <td class="tg-8d8j">89,57</td>
    <td class="tg-2b7s">88.28(87.28-89.26)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE--</td>
    <td class="tg-2b7s">82.45</td>
    <td class="tg-8d8j">74.16</td>
    <td class="tg-8d8j">88.21</td>
    <td class="tg-2b7s">88.41(87.54-89.26)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE</td>
    <td class="tg-2b7s">82.62</td>
    <td class="tg-8d8j">74.28</td>
    <td class="tg-8d8j">88.42</td>
    <td class="tg-2b7s">88.5o(87.46-89.57)</td>
  </tr>
  <tr>
    <td class="tg-2b7s" rowspan="5">RcsNet-50</td>
    <td class="tg-7zrl">Scratch</td>
    <td class="tg-2b7s">80.50</td>
    <td class="tg-8d8j">65.42</td>
    <td class="tg-8d8j">90.97</td>
    <td class="tg-2b7s">86.22(85.22-87.35)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ImngeNet</td>
    <td class="tg-2b7s">81.73</td>
    <td class="tg-8d8j">68.36</td>
    <td class="tg-8d8j">91.01</td>
    <td class="tg-2b7s">87.87(86.85-88.85)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MD-MoCo</td>
    <td class="tg-2b7s">82.35</td>
    <td class="tg-8d8j">73.12</td>
    <td class="tg-8d8j">88.76</td>
    <td class="tg-2b7s">87.89(87.06-88.88)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE--</td>
    <td class="tg-2b7s">81.10</td>
    <td class="tg-8d8j">69.03</td>
    <td class="tg-8d8j">89.48</td>
    <td class="tg-2b7s">87.14(86.10-88.22)</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE</td>
    <td class="tg-2b7s">82.60</td>
    <td class="tg-8d8j">74.53</td>
    <td class="tg-8d8j">88.21</td>
    <td class="tg-2b7s">88.37(87.38-89.32)</td>
  </tr>
</tbody>
</table>
</div>


![image](https://user-images.githubusercontent.com/50255927/189317679-e3c22309-899b-4f8f-a689-d81e406376b5.png)

### Results for various tasks
+ Note: **segmentation task** for Lung and **detection task** for TBX

<table class="tg">
<thead>
  <tr>
    <th class="tg-7zrl" rowspan="2">Backbones</th>
    <th class="tg-7zrl" rowspan="2">Pre-train</th>
    <th class="tg-8d8j" colspan="2">Lung</th>
    <th class="tg-8d8j" colspan="3">TBX</th>
  </tr>
  <tr>
    <th class="tg-2b7s">Dice</th>
    <th class="tg-7zrl">mloU</th>
    <th class="tg-7zrl">mAP</th>
    <th class="tg-7zrl">AP-Active</th>
    <th class="tg-7zrl">AP-Latent</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7zrl" rowspan="5">ResNet-18</td>
    <td class="tg-7zrl">Scratch</td>
    <td class="tg-2b7s">95.24</td>
    <td class="tg-2b7s">94.00</td>
    <td class="tg-2b7s">30.71</td>
    <td class="tg-2b7s">56.71</td>
    <td class="tg-2b7s">4.72</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ImageNet</td>
    <td class="tg-2b7s">95.26</td>
    <td class="tg-2b7s">94.10</td>
    <td class="tg-2b7s">29.46</td>
    <td class="tg-2b7s">56.27</td>
    <td class="tg-2b7s">2.66</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MD-MoCo</td>
    <td class="tg-2b7s">95.31</td>
    <td class="tg-2b7s">94.14</td>
    <td class="tg-2b7s">36.00</td>
    <td class="tg-2b7s">67.17</td>
    <td class="tg-2b7s">4.84</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE--</td>
    <td class="tg-2b7s">95.14</td>
    <td class="tg-2b7s">93.90</td>
    <td class="tg-2b7s">34.70</td>
    <td class="tg-2b7s">63.43</td>
    <td class="tg-2b7s">5.97</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE</td>
    <td class="tg-2b7s">95.37</td>
    <td class="tg-2b7s">94.22</td>
    <td class="tg-2b7s">36.71</td>
    <td class="tg-2b7s">64.84</td>
    <td class="tg-2b7s">8.59</td>
  </tr>
  <tr>
    <td class="tg-7zrl" rowspan="5">　<br>ResNet-50</td>
    <td class="tg-7zrl">Scratch</td>
    <td class="tg-2b7s">93.52</td>
    <td class="tg-2b7s">92.03</td>
    <td class="tg-2b7s">23.93</td>
    <td class="tg-2b7s">44.85</td>
    <td class="tg-2b7s">3.01</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ImageNet</td>
    <td class="tg-2b7s">93.77</td>
    <td class="tg-2b7s">92.43</td>
    <td class="tg-2b7s">35.61</td>
    <td class="tg-2b7s">58.81</td>
    <td class="tg-2b7s">12.42</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MD-MoCo</td>
    <td class="tg-2b7s">94.33</td>
    <td class="tg-2b7s">93.04</td>
    <td class="tg-2b7s">36.78</td>
    <td class="tg-2b7s">64.37</td>
    <td class="tg-2b7s">9.18</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE--</td>
    <td class="tg-2b7s">95.04</td>
    <td class="tg-2b7s">93.82</td>
    <td class="tg-2b7s">35.14</td>
    <td class="tg-2b7s">57.32</td>
    <td class="tg-2b7s">12.97</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MUSCLE</td>
    <td class="tg-2b7s">95.27</td>
    <td class="tg-2b7s">94.10</td>
    <td class="tg-2b7s">37.83</td>
    <td class="tg-2b7s">63.46</td>
    <td class="tg-2b7s">12.21</td>
  </tr>
</tbody>
</table>

![image](https://user-images.githubusercontent.com/50255927/189317479-14ecb3de-da80-4df3-b9a0-f1fece7b953f.png)

## Citation
***
If our work is helpful to you, please kindly cite our paper as:
```
@inproceedings{liao2022muscle,  
title={MUSCLE: Multi-task Self-supervised Continual Learning to Pre-train Deep Models for X-ray Images of Multiple Body Parts},  
author={Weibin, Liao and Haoyi, Xiong and Qingzhong, Wang and Yan, Mo and Xuhong, Li and Yi, Liu and Zeyu, Chen and Siyu, Huang and Dejing, Dou},  
booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
year={2022},  
organization={Springer}  
}  
```

