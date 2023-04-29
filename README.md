# Generative and Unsupervised Deep Learning @ KAIST

## Course Information
**Instructor:** Sung Ju Hwang (sjhwang82@kaist.ac.kr)  
**TAs:** Seul Lee (animecult@kaist.ac.kr), Geon Park and Sohyun An 

**Office:** 
This is an on/offline hybrid course.
Building Nubmer 9, Room 9201 (Instructor) 2nd floor (TAs)  
Office hours: By appointment only.

### Grading Policy
* **Absolute Grading**
* Paper Presentation: 25%
* Attendance and Participation: 25%
* Assignments and Project: 50%

## Tentative Schedule

| Dates | Topic | 
|---|:---|
|2/28| Course Introduction |
|3/2| Autoencoders and Variational Autoencoders (Lecture) |
|3/7| Transformers for Language and Vision (Lecture) |
|3/9| Transformers for Language and Vision (Lecture) |
|3/14| Self-Supervised Learning (Lecture) **Review Due** |
|3/16| Self-Supervised Learning (Lecture) | 
|3/21| Self-Supervised Learning (Presentation) | 
|3/23| Advanced VAEs and GANs (Lecture) **Review Due** |
|3/28| Advanced VAEs and GANs (Lecture) **Review Due** |
|3/30| Advanced VAEs and GANs (Presentation) |
|4/4| **VAEs and GANs - VQVAE and VQGAN (Lab session), initial proposal due April 2nd**  |
|4/6| Autoregressive and Flow-based Models (Lecture) | 
|4/11| Diffusion Models  (Lecture) **Review Due, Presentation Slides Due (Diffusion Models)** |
|4/13| Diffusion Models  (Lecture) |
|4/18| Diffusion Models (Presentation) |
|4/20| **Mid-term Presentation, Presentation Slides Due (Large Language Models)** | 
|4/25| Large Language Models (Lecture) **Review Due** |
|4/27| Large Language Models (Presentation) **Presentation Slides Due (Multimodal Foundation Models)**|
|5/2| Multimodal Foundation Models (Lecture) **Review Due** | 
|5/4| Multimodal Foundation Models (Presentation) **Presentation Slides Due (Text-to-Image Generation)**| 
|5/9| Text-to-Image Generation (Lecture) |
|5/11| **Text-to-Image Generation - LDM (Lab Session)** |
|5/16| Text-to-Image Generation (Presentation) |
|5/23| Graph Representation Learning (Lecture) **Review Due, Presentation Slides Due (Graph Reprsentation Learning and Generation)** | 
|5/25| Graph Generation (Lecture) | 
|6/1| Graph Generation (Presentation) |
|6/6| **Molecular Graph Generation - GDSS, MOOD, DruM (Lab session), Presentation Slides Due (Speech Synthesis)** |
|6/8| Speech Synthesis (Lecture) **Review Due** |
|6/13| Speech Synthesis (Presentation) **Final report due** |  
|6/15| **Final Presentation**

## Reading List

### Transformers and Vision Transformers
[[Vaswani et al. 17]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) Attention is All You Need, NeurIPS 2017.  
[[Beltagy et al. 20]](https://arxiv.org/abs/2004.05150) Longformer: The Long-Document Transformer, arXiv preprint, 2020.  
[[Zaheer et al. 20]](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf) Big Bird: Transformers for Longer Sequences, NeurIPS 2020.  
[[Dosovitskiy et al. 21]](https://openreview.net/forum?id=YicbFdNTTy) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021.  
[[Touvron et al. 21]](http://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf) Training Data-efficient Image transformers & Distillation through Attention, ICML 2021.  
[[Tay et al. 21]](http://proceedings.mlr.press/v139/tay21a/tay21a.pdf) Synthesizer: Rethinking Self-Attention for Transformer Models, ICML 2021.  
[[Liu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, ICCV 2021.  
[[Wu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf) CvT: Introducing Convolutions to Vision Transformers, ICCV 2021.  
[[Dai et al. 21]](https://proceedings.neurips.cc/paper/2021/file/20568692db622456cc42a2e853ca21f8-Paper.pdf) CoAtNet: Marrying Convolution and Attnetion for All Data Sizes, NeurIPS 2021.  
[[Yang et al. 21]](https://proceedings.neurips.cc/paper/2021/file/fc1a36821b02abbd2503fd949bfc9131-Paper.pdf) Focal Attention for Long-Range Interactions in Vision Transformers, NeurIPS 2021.  
[[Rao et al. 21]](https://openreview.net/pdf?id=jB0Nlbwlybm) DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification, NeurIPS 2021.  
[[El-Nouby et al. 21]](https://proceedings.neurips.cc/paper/2021/file/a655fbe4b8d7439994aa37ddad80de56-Paper.pdf) XCiT: Cross-Covariance Image Transformers, NeurIPS 2021.  
[[Li et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.pdf) MViTv2: Improved Multiscale Vision Transformers for Classification and Detection, CVPR 2022.  
[[Lee et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_MPViT_Multi-Path_Vision_Transformer_for_Dense_Prediction_CVPR_2022_paper.pdf) MPViT : Multi-Path Vision Transformer for Dense Prediction, CVPR 2022.  
***
[[Lee et al. 23]](https://openreview.net/forum?id=VV0hSE8AxCw) Sparse Token Transformer with Attention Back Tracking, ICLR 2023.  
[[Liu et al. 23]](https://openreview.net/pdf?id=De4FYqjFueZ) Transformers Learn Shortcuts to Automata, ICLR 2023.  
[[Bolya et al. 23]](https://openreview.net/pdf?id=De4FYqjFueZ) Token Merging: Your ViT But Faster, ICLR 2023.  
***

### Self-Supervised Learning
<!--
[[Oliver et al. 18]](https://papers.nips.cc/paper/7585-realistic-evaluation-of-deep-semi-supervised-learning-algorithms.pdf) Realistic Evaluation of Deep Semi-Supervised Learning Algorithms, NeurIPS 2018.  
[[Berthelot et al. 19]](https://papers.nips.cc/paper/8749-mixmatch-a-holistic-approach-to-semi-supervised-learning.pdf) MixMatch: A Holistic Approach to Semi-Supervised Learning, NeurIPS 2019.  
[[Sohn et al. 20]](https://arxiv.org/pdf/2001.07685.pdf) Fixmatch: Simplifying semi-supervised learning with consistency and confidence, arXiv prepring, 2020.--> 
[[Dosovitskiy et al. 14]](https://papers.nips.cc/paper/5548-discriminative-unsupervised-feature-learning-with-convolutional-neural-networks.pdf) Discriminative Unsupervised Feature Learning with Convolutional Neural Networks, NIPS 2014.  
[[Pathak et al. 16]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf) Context Encoders: Feature Learning by Inpainting, CVPR 2016.  
[[Norrozi and Favaro et al. 16]](https://arxiv.org/pdf/1603.09246.pdf) Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles, ECCV 2016.   
[[Gidaris et al. 18]](https://openreview.net/pdf?id=S1v4N2l0-) Unsupervised Representation Learning by Predicting Image Rotations, ICLR 2018.  
[[He et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020.  
[[Chen et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/6165-Paper.pdf) A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020.  
[[Mikolov et al. 13]](https://arxiv.org/pdf/1301.3781.pdf) Efficient Estimation of Word Representations in Vector Space, ICLR 2013.  
[[Devlin et al. 19]](https://www.aclweb.org/anthology/N19-1423.pdf) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019.  
[[Clark et al. 20]](https://openreview.net/pdf?id=r1xMH1BtvB) ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators, ICLR 2020.  
[[Hu et al. 20]](https://openreview.net/forum?id=HJlWWJSFDH) Strategies for Pre-training Graph Neural Networks, ICLR 2020.  
[[Chen et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/6022-Paper.pdf) Generative Pretraining from Pixels, ICML 2020.  
[[Laskin et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/5951-Paper.pdf) CURL: Contrastive Unsupervised Representations for Reinforcement Learning, ICML 2020.  
[[Grill et al. 20]](https://arxiv.org/pdf/2006.07733.pdf) Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning, NeurIPS 2020.    
[[Chen et al. 20]](https://arxiv.org/pdf/2006.10029v1.pdf) Big Self-Supervised Models are Strong Semi-Supervised Learners, NeurIPS, 2020.   
[[Chen and He. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf) Exploring Simple Siamese Representation Learning, CVPR 2021.  
[[Tian et al. 21]](https://arxiv.org/pdf/2102.06810.pdf) Understanding Self-Supervised Learning Dynamics without Contrastive Pairs, ICML 2021.  
[[Caron et al. 21]](https://arxiv.org/pdf/2104.14294.pdf) Emerging Properties in Self-Supervised Vision Transformers, ICCV 2021.  
[[Bao et al. 22]](https://openreview.net/pdf?id=p-BhZSz59o4) BEiT: BERT Pre-Training of Image Transformers, ICLR 2022.  
[[Bardes et al. 22]](https://openreview.net/forum?id=xm6YD62D1Ub) VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, ICLR 2022.  
[[He et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) Masked Autoencoders are Scalable Vision Learners, CVPR 2022.  
[[Liu et al. 22]](https://arxiv.org/pdf/2203.15508.pdf) Improving Contrastive Learning with Model Augmetnation, arXiv preprint, 2022.  
***
[[Touvron et al. 22]](https://arxiv.org/abs/2204.07118) DeIT III: Revenge of the VIT, ECCV 2022.  
[[Garrido. et al. 23]](https://openreview.net/forum?id=kDEL91Dufpa) On the duality between contrastive and non-contrastive self-supervised learning, ICLR 2023.   
[[Lee et al. 23]](https://openreview.net/forum?id=kIAx30hYi_p) Self-Supervised Set Representation Learning for Unsupervised Meta-Learning, ICLR 2023.  
[[Park et al. 23]](https://openreview.net/forum?id=azCKuYyS74) What Do Self-Supervised Vision Transformers Learn?, ICLR 2023.  
***

### 

### Variational Autoencoders, Autoregressive and Flow-Based Generative Models
[[Kingma and Welling 14]](https://arxiv.org/pdf/1312.6114.pdf) Auto-Encoding Variational Bayes, ICLR 2014.   
[[Sohn et al. 15]](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf) Learning Structured Output Representation
using Deep Conditional Generative Model, NeurIPS 2015.  
[[Higgins et al. 17]](https://openreview.net/forum?id=Sy2fzU9gl) beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, ICLR 2017.  
[[van den Oord et al. 17]](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf) Neural Discrete Representation Learning, NeurIPS 2017.  
[[Razavi et al. 19]](https://proceedings.neurips.cc/paper/2019/file/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Paper.pdf) Generating Diverse High-Fidelity Images with VQ-VAE-2, NeurIPS 2019.    
[[Vahdat and Kautz 20]](https://arxiv.org/pdf/2007.03898v1.pdf) NVAE: A Deep Hierarchical Variational Autoencoder, NeurIPS 2020.  
[[Rezende and Mohamed 15]](http://proceedings.mlr.press/v37/rezende15.pdf) Variational Inference with Normalizing Flows, ICML 2015.   
[[Germain et al. 15]](http://proceedings.mlr.press/v37/germain15.pdf) MADE: Masked Autoencoder for Distribution Estimation, ICML 2015.  
[[Kingma et al. 16]](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow.pdf) Improved Variational Inference with Inverse Autoregressive Flow, NeurIPS 2016.  
[[Oord et al. 16]](http://proceedings.mlr.press/v48/oord16.pdf) Pixel Recurrent Neural Networks, ICML 2016.  
[[Oord et al. 16]](https://proceedings.neurips.cc/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf) Conditional Image Generation with PixelCNN Decoders, NeurIPS 2016.  
[[Dinh et al. 17]](https://openreview.net/pdf?id=HkpbnH9lx) Density Estimation Using Real NVP, ICLR 2017.  
[[Papamakarios et al. 17]](https://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation.pdf) Masked Autoregressive Flow for Density Estimation, NIPS 2017.  
[[Huang et al.18]](http://proceedings.mlr.press/v80/huang18d/huang18d.pdf) Neural Autoregressive Flows, ICML 2018.  
[[Kingma and Dhariwal 18]](http://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions.pdf) Glow: Generative Flow with Invertible 1x1 Convolutions, NeurIPS 2018.  
[[Ho et al. 19]](http://proceedings.mlr.press/v97/ho19a/ho19a.pdf) Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design, ICML 2019.    
[[Chen et al. 19]](https://papers.nips.cc/paper/9183-residual-flows-for-invertible-generative-modeling.pdf) Residual Flows for Invertible Generative Modeling, NeurIPS 2019.  
[[Tran et al. 19]](https://papers.nips.cc/paper/9612-discrete-flows-invertible-generative-models-of-discrete-data.pdf) Discrete Flows: Invertible Generative
Models of Discrete Data, NeurIPS 2019.  
[[Ping et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/647-Paper.pdf) WaveFlow: A Compact Flow-based Model for Raw Audio, ICML 2020.  
[[Chang et al. 22]](https://arxiv.org/abs/2202.04200) MaskGIT: Masked Generative Image Transformer, CVPR 2022.  
***

[[Chen et al. 22]](https://openreview.net/forum?id=zrAUoI2JA2) Learning Continuous Normalizing Flows for Faster Converegence to Target Distribution via Ascent Regularizations, ICLR 2023.  
[[Lipman et al. 23]](https://openreview.net/pdf?id=PqvMRDCJT9t) Flow Matching for Generative Modeling, ICLR 2023.  

***

### Generative Adversarial Networks
[[Goodfellow et al. 14]](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) Generative Adversarial Nets, NIPS 2014.   
[[Radford et al. 15]](https://arxiv.org/abs/1511.06434) Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016.   
[[Chen et al. 16]](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf) InfoGAN: Interpreting Representation Learning by Information Maximizing Generative Adversarial Nets, NIPS 2016.   
[[Arjovsky et al. 17]](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) Wasserstein Generative Adversarial Networks, ICML 2017.  
[[Zhu et al. 17]](https://arxiv.org/pdf/1703.10593.pdf) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017.  
[[Zhang et al. 17]](https://arxiv.org/pdf/1706.03850.pdf) Adversarial Feature Matching for Text Generation, ICML 2017.  
[[Karras et al. 18]](https://openreview.net/forum?id=Hk99zCeAb) Progressive Growing of GANs for Improved Quality, Stability, and Variation, ICLR 2018.  
[[Choi et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf) StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation, CVPR 2018.  
[[Brock et al. 19]](https://openreview.net/pdf?id=B1xsqj09Fm) Large Scale GAN Training for High-Fidelity Natural Image Synthesis, ICLR 2019.  
[[Karras et al. 19]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf) A Style-Based Generator Architecture for Generative Adversarial Networks, CVPR 2019.  
[[Karras et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf) Analyzing and Improving the Image Quality of StyleGAN, CVPR 2020.  
[[Sinha et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/1324-Paper.pdf) Small-GAN: Speeding up GAN Training using Core-Sets, ICML 2020.  
[[Karras et al. 20]](https://papers.nips.cc/paper/2020/file/8d30aa96e72440759f74bd2306c1fa3d-Paper.pdf) Training Generative Adversarial Networks with
Limited Data, NeurIPS 2020.  
[[Liu et al. 21]](https://openreview.net/pdf?id=1Fqg133qRaI) Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis, ICLR 2021.  
[[Esser et al. 22]](https://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf) Taming Transformers for High-Resolution Image Synthesis, CVPR 2021.  
[[Hudson and Zitnick 21]](https://arxiv.org/pdf/2103.01209.pdf) Generative Adversarial Transformers, ICML 2021.  
[[Karras et al. 21]](https://proceedings.neurips.cc/paper/2021/file/076ccd93ad68be51f23707988e934906-Paper.pdf) Alias-Free Generative Adversarial Networks, NeurIPS 2021.  
[[Lin et al. 22]](https://openreview.net/pdf?id=ufGMqIM0a4b) InfinityGAN: Towards Infinite-Pixel Image Synthesis, ICLR 2022.  
[[Lee et al. 22]](https://openreview.net/pdf?id=dwg5rXg1WS_) ViTGAN: Training GANs with Vision Transformers, ICLR 2022.  
[[Yu et al. 22]](https://openreview.net/forum?id=pfNyExj7z2) Vector-Quantized Image Modeling with Improved VQGAN, ICLR 2022. 
***
[[Huang et al. 22]](https://openreview.net/forum?id=js2ssA77fX) Masked Generative Adversarial Networks are Data-Efficient Generation Learners, NeurIPS 2022.  
[[Yang et al. 22]](https://openreview.net/pdf?id=_P4JCoz83Mb) Distilling Representations from GAN Generator via Squeeze and Span, NeurIPS 2022.  
[[Brooks et al. 22]](https://openreview.net/pdf?id=VnAwNNJiwDb) Generating Long Videos of Dynamic Scenes, NeurIPS 2022.  
[[Wang et al. 23]](https://openreview.net/forum?id=HZf7UbpWHuA) Diffusion-GAN: Training GANs with Diffusion, ICLR 2023.  

***

### Diffusion Models
[[Sohl-Dickstein et al. 15]](https://proceedings.mlr.press/v37/sohl-dickstein15.html) Deep Unsupervised Learning using Nonequilibrium Thermodynamics, ICML 2015.  
[[Song and Ermon 19]](https://proceedings.neurips.cc/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf) Generative Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2019.  
[[Song and Ermon 20]](https://papers.nips.cc/paper/2020/file/92c3b916311a5517d9290576e3ea37ad-Paper.pdf) Improved Techniques for Training Score-Based Generative Models, NeurIPS 2020.  
[[Ho et al. 20]](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) Denoising Diffusion Probabilistic Models, NeurIPS 2020.  
[[Song et al. 21]](https://openreview.net/forum?id=PxTIG12RRHS) Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 2021.  
[[Nichol and Dhariwal 21]](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) Improved Denoising Diffusion Probabilistic Models, ICML 2021.  
[[Vahdat et al. 21]](https://proceedings.neurips.cc/paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Paper.pdf) Score-based Generative Modeling in Latent Space, NeurIPS 2021.  
[[Dhariwal and Nichol 21]](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf) Diffusion Models Beat GANs on Image Synthesis, NeureIPS 2021.  
[[De Bortoli et al. 22]](https://proceedings.neurips.cc/paper/2021/file/940392f5f32a7ade1cc201767cf83e31-Paper.pdf) Diffusion Schrodinger Bridge with Application to Score-Based Generative Modeling, NeurIPS 2021.  
[[Ho and Salimans 22]](https://arxiv.org/pdf/2207.12598.pdf) Classifier-Free Diffusion Guidance, arXiv preprint, 2022.  
[[Dockhorn et al. 22]](https://openreview.net/forum?id=CzceR82CYc) Score-Based Generative Modeling with Critically-Damped Langevin Diffusion, ICLR 2022.  
[[Salimans and Ho 22]](https://openreview.net/pdf?id=TIdIXIpzhoI) Progressive Distillation for Fast Sampling of Diffusion Models, ICLR 2022.  
[[Chen et al. 22]](https://openreview.net/forum?id=nioAdKCEdXB) Likelihood Training of Schrodinger Bridge using Forward-Backwrad SDEs Theory, ICLR 2022.  
***
[[Cohen et al. 22]](https://proceedings.mlr.press/v162/cohen22b/cohen22b.pdf) Diffusion bridges vector quantized variational autoencoders, ICML 2022.  
[[Ho et al. 22]](https://openreview.net/forum?id=f3zNgKga_ep) Video Diffusion Models, NeurIPS 2022.  
[[Chen et al. 23]](https://openreview.net/forum?id=zyLVMgsZ0U_) Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions, ICLR 2023.   
[[Liu et al. 23]](https://openreview.net/forum?id=WH1yCa0TbB) Learning Diffusion Bridges on Constrained Domains, ICLR 2023.   
[[Chung et al. 23]](https://openreview.net/forum?id=OnD9zGAGT0k) Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 2023.  
[[Chen et al. 23]](https://arxiv.org/abs/2211.06956) Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding, CVPR 2023.  
[[Song et al. 23]](https://arxiv.org/abs/2303.01469) Consistency Models, arXiv preprint 2023.  
***

### Large Language Models
[[Shoeybi et al. 19]](https://arxiv.org/abs/1909.08053) Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism, arXiv preprint, 2019.  
[[Lewis et al. 20]](https://aclanthology.org/2020.acl-main.703.pdf) BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension, ACL 2020.  
[[Raffel et al. 20]](https://jmlr.org/papers/volume21/20-074/20-074.pdf) Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, JMLR 2020.  
[[Gururangan et al. 20]](https://aclanthology.org/2020.acl-main.740.pdf) Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks, ACL 2020.  
[[Brown et al. 20]](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) Language Models are Few-shot Learners, NeurIPS 2020.  
[[Rae et al. 21]](https://arxiv.org/abs/2112.11446) Scaling Language Models: Methods, Analysis & Insights from Training Gopher, arXiv preprint, 2021.  
[[Thoppilan et al. 22]](https://arxiv.org/pdf/2201.11903.pdf) LaMDA: Language Models for Dialog Applications, arXiv preprint, 2022.   
[[Wei et al. 22]](https://openreview.net/pdf?id=gEZrGCozdqR) Finetuned Langauge Models Are Zero-Shot Learners, ICLR 2022.  
[[Wang et al. 22]](https://openreview.net/forum?id=pMQwKL1yctf) Language Modeling via Stochastic Processes, ICLR 2022.  
[[Alayrac et al. 22]](https://arxiv.org/abs/2204.14198) Flamingo: a Visual Language Model for Few-Shot Learning, arXiv preprint, 2022.  
[[Chowdhery et al. 22]](https://arxiv.org/abs/2204.02311) PaLM: Scaling Langauge Modeling with Pathways, arXiv preprint, 2022.  
[[Wei et al. 22]](https://arxiv.org/pdf/2201.11903.pdf) Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, NeurIPS 2022.  
[[Touvron et al. 23]](https://research.facebook.com/file/1574548786327032/LLaMA--Open-and-Efficient-Foundation-Language-Models.pdf) LLaMA: Open and Efficient Foundation Language Models, arXiv preprint, 2023.  
***
[[Ouyang et al. 22]](https://openreview.net/forum?id=TG8KACxEON) Training Language Models to Follow Instructions with Human Feedback, NeurIPS 2022.  
[[Wang et al. 23]](https://openreview.net/forum?id=1PL1NIMMrw) Self-Consistency Improves Chain of Thought Reasoning in Language Models, ICLR 2023.  
[[Rust et al. 23]](https://openreview.net/forum?id=FkSp8VW8RjH) Language Modelling with Pixels, ICLR 2023.  
[[Arora et al. 23]](https://openreview.net/pdf?id=bhUPJnS2g0X) Ask Me Anything: A Simple Strategy for Prompting Langauge Models, ICLR 2023.  
[[Honovich et al. 22]](https://arxiv.org/abs/2212.09689) Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor, arXiv preprint, 2022.  
[[Wang et al. 22]](https://arxiv.org/abs/2212.10560) Self-Instruct: Aligning Language Model with Self Generated Instructions, arXiv preprint, 2022.  
***

### Multimodal Foundation Models 
[[Socher et al. 13]](https://papers.nips.cc/paper/2013/file/2d6cc4b2d139a53512fb8cbb3086ae2e-Paper.pdf) Zero-Shot Learning Through Cross-Modal Transfer, NeurIPS 2013.  
[[Lu et al. 19]](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf) VilBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks, NeurIPS 2019.  
[[Huang et al. 20]](https://arxiv.org/pdf/2004.00849.pdf) Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers, arXiv preprint 2020.  
[[Li et al. 20]](https://ojs.aaai.org/index.php/AAAI/article/view/6795) Unicoder-VL: A Universal Encoder for Vision and Language by Cross-Modal Pre-Training, AAAI 2020.  
[[Radford et al. 21]](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) Learning Transferable Visual Models From Natural Language Supervision, ICML 2021.  
[[Singh et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Singh_FLAVA_A_Foundational_Language_and_Vision_Alignment_Model_CVPR_2022_paper.pdf) FLAVA: A Foundational Languageand Vision Asignment Model, CVPR 2022.  
[[Li et al. 22]](https://proceedings.mlr.press/v162/li22n/li22n.pdf) BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation, ICML 2022.  
[[Baevski et al. 22]](https://proceedings.mlr.press/v162/baevski22a/baevski22a.pdf) data2vec: A General Framework for Self-supervised Learning in Speech, Vision, and Language, ICML 2022.  
[[Fei et al. 22]](https://www.nature.com/articles/s41467-022-30761-2) Towards artificial general intelligence via a multimodal foundation model, Nature Communications 2022.  
***
[[Alayract et al. 22]](https://openreview.net/forum?id=EbMuimAbPbs) Flamingo: a Visual Language Model for Few-shot Learning, NeurIPS 2022.   
[[Wang et al. 22]](https://arxiv.org/abs/2208.10442) Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks, arXiv preprint, 2022.  
[[Reed et al. 22]](https://arxiv.org/pdf/2205.06175.pdf) A Generalist Agent, arXiv preprint, 2022.  
[[Zeng et al. 23]](https://openreview.net/forum?id=G2Q2Mh3avow) Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language, ICLR 2023.  

***

### Text-to-Image Synthesis
[[Reed et al. 16]](http://proceedings.mlr.press/v48/reed16.pdf) Generative Adversarial Text to Image Synthesis, ICML 2016.  
[[Li et al. 19]](https://proceedings.neurips.cc/paper/2019/file/1d72310edc006dadf2190caad5802983-Paper.pdf) Controllable Text-to-Image Generation, NeurIPS 2019.  
[[Ramesh et al. 21]](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf) Zero-Shot Text-to-Image Generation, ICML 2021.   
[[Radford et al. 21]](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) Learning Transferable Visual Models From Natural Language Supervision, ICML 2021.  
[[Ding et al. 21]](https://proceedings.neurips.cc/paper/2021/file/a4d92e2cd541fca87e4620aba658316d-Paper.pdf) CogView: Mastering Text-to-Image Generation via Transformers, NeurIPS 2021.  
[[Zou et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Towards_Language-Free_Training_for_Text-to-Image_Generation_CVPR_2022_paper.pdf) Towards Language-Free Training for Text-to-Image Generation, CVPR 2022.  
[[Rombach et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) High-Resolution Image Synthesis with Latent Diffusion Models, CVPR 2022.  
[[Gu et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf) Vector Quantized Diffusion Model for Text-to-Image Synthesis, CVPR 2022.  
[[Nichol et al. 22]](https://proceedings.mlr.press/v162/nichol22a/nichol22a.pdf) GLIDE: Towards Photorealistic Image Generation and Editing with
Text-Guided Diffusion Models, ICML 2022.  
***
[[Saharia et al. 22]](https://arxiv.org/abs/2205.11487) Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding, arXiv preprint, 2022.  
[[Yu et al. 22]](https://arxiv.org/abs/2206.10789) Scaling Autoregressive Models for Content-Rich Text-to-Image Generation, arXiv preprint, 2022.  
[[Gafni et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750087.pdf) Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors, ECCV 2022.  
[[Chen et al. 23]](https://openreview.net/forum?id=XSEBx0iSjFQ) Re-Imagen: Retrieval-Augmented Text-to-Image Generator, ICLR 2023.  
[[Poole et al. 23]](https://openreview.net/forum?id=FjNys5c7VyY) DreamFusion: Text-to-3D using 2D Diffusion, ICLR 2023.  
[[Chang et al. 23]](https://arxiv.org/abs/2301.00704) Muse: Text-To-Image Generation via Masked Generative Transformers, arXiv preprint, 2023.  
***

### Speech Representation Learning and Synthesis
[[Oord et al. 16]](https://arxiv.org/abs/1609.03499) WaveNet: A Generative Model for Raw Audio, arXiv preprint 2016.  
[[Baevski et al. 20]](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf) wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations, NeurIPS 2020.  
[[Tang et al. 22]](https://aclanthology.org/2022.acl-long.105.pdf) Unified Speech-Text Pre-training for Speech Translation and Recognition, ACL 2022.  
[[Wang et al. 17]](https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/1452.PDF) Tacotron: Towards End-to-End Speech Synthesis, Interspeech 2017.  
[[Shen et al. 18]](https://arxiv.org/abs/1712.05884) Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions, ICASSP 2018.  
[[Chen et al. 19]](https://openreview.net/forum?id=rkzjUoAcFX) Sample Efficient Adaptive Text-to-Speech, ICLR 2019.  
[[Hsu et al. 19]](https://openreview.net/forum?id=rygkk305YQ) Hierarchical Generative Modeling for Controllable Speech Synthesis, ICLR 2019.  
[[Kumar et al. 19]](https://papers.nips.cc/paper/2019/file/6804c9bca0a615bdb9374d00a9fcba59-Paper.pdf) MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis, NeurIPS 2019.  
[[Kong et al. 20]](https://papers.nips.cc/paper/2020/file/c5d736809766d46260d816d8dbc9eb44-Paper.pdf) HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis, NeurIPS 2020.  
[[Min et al. 21]](https://openreview.net/pdf?id=De4FYqjFueZ) Meta-StyleSpeech: Multi-Speaker Adaptive Text-to-Speech Generation, ICML 2021.  
[[Tang et al. 22]](https://aclanthology.org/2022.acl-long.105.pdf) Unified Speech-Text Pre-training for Speech Translation and Recognition, ACL 2022.  
***
[[Hsu and Shi 22]](https://openreview.net/forum?id=zrAUoI2JA2) u-HuBERT: Unified Mixed-Modal Speech Pretraining and Zero-Shot Transfer to Unlabeled Modality, NeurIPS 2022.  
[[Kang et al. 23]](https://arxiv.org/abs/2211.09383) Any-Speaker Adaptive Text-To-Speech Synthesis with Diffusion Models, ICASSP 2023.  
[[Ren et al. 23]](https://arxiv.org/abs/2211.09383) Back of Tricks for Unsupervised Text-to-Speech, ICLR 2023.    
[[Lee et al. 23]](https://openreview.net/forum?id=zrAUoI2JA2) BigVGAN: A Universal Neural Vocoder with Large-Scale Training, ICLR 2023.  
***


### Graph Representation Learning and Generation
[[Li et al. 16]](https://arxiv.org/pdf/1511.05493.pdf) Gated Graph Sequence Neural Networks, ICLR 2016.  
[[Hamilton et al. 17]](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf) Inductive Representation Learning on Large Graphs, NIPS 2017.  
[[Kipf and Welling 17]](https://openreview.net/pdf?id=SJU4ayYgl) Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017.  
[[Velickovic et al. 18]](https://openreview.net/pdf?id=rJXMpikCZ) Graph Attention Networks, ICLR 2018.   
[[Ying et al. 18]](https://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf) Hierarchical Graph Representation Learning with Differentiable Pooling, NeurIPS 2018.  
[[Xu et al. 19]](https://openreview.net/forum?id=ryGs6iA5Km) How Powerful are Graph Neural Networks?, ICLR 2019.  
[[Maron et al. 19]](https://papers.nips.cc/paper/8488-provably-powerful-graph-networks.pdf) Provably Powerful Graph Networks, NeurIPS 2019.  
[[Yun et al. 19]](https://papers.nips.cc/paper/9367-graph-transformer-networks.pdf) Graph Transformer Neteworks, NeurIPS 2019.  
[[Loukas 20]](https://arxiv.org/pdf/1907.03199.pdf) What Graph Neural Networks Cannot Learn: Depth vs Width, ICLR 2020.  
[[Bianchi et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/1614-Paper.pdf) Spectral Clustering with Graph Neural Networks for Graph Pooling, ICML 2020.  
[[Xhonneux et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/4075-Paper.pdf) Continuous Graph Neural Networks, ICML 2020.  
[[Garg et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/2911-Paper.pdf) Generalization and Representational Limits of Graph Neural Networks, ICML 2020.  
[[Baek et al. 21]](https://openreview.net/forum?id=JHcqXGaqiGn) Accurate Learning of Graph Representations with Graph Multiset Pooling, ICLR 2021.  
[[Liu et al. 21]](http://proceedings.mlr.press/v139/liu21k/liu21k.pdf) Elastic Graph Neural Networks, ICML 2021.  
[[Li et al. 21]](http://proceedings.mlr.press/v139/li21o/li21o.pdf) Training Graph Neural networks with 1000 Layers, ICML 2021.  
[[Jo et al. 21]](https://openreview.net/forum?id=vwgsqRorzz) Edge Representation Learning with Hypergraphs, NeurIPS 2021.  
[[Guo et al. 22]](https://openreview.net/pdf?id=l4IHywGq6a) Data-Efficient Graph Grammar Learning for Molecular Generation, ICLR 2022.  
[[Geerts et al. 22]](https://openreview.net/pdf?id=wIzUeM3TAU) Expressiveness and Approximation Properties of Graph Neural Networks, ICLR 2022.  
[[Bevilacqua et al. 22]](https://openreview.net/pdf?id=dFbKQaRk15w) Equivariant Subgraph Aggregation Networks, ICLR 2022.  
[[Jo et al. 22]](https://proceedings.mlr.press/v162/jo22a/jo22a.pdf) Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations, ICML 2022.  
[[Hoogeboom et al. 22]](https://proceedings.mlr.press/v162/hoogeboom22a/hoogeboom22a.pdf) Equivariant Diffusion for Molecule Generation in 3D, ICML 2022.  
***
[[Kim et al. 22]](https://openreview.net/forum?id=JY6fLgR8Yq) Graph Self-Supervised Learning with Accurate Discrepancy Learning, NeurIPS 2022.  
[[Kim et al. 23]](https://openreview.net/forum?id=um2BxfgkT2_) Pure Transformers are Powerful Graph Learners, NeurIPS 2022.  
[[Vignac et al. 23]](https://openreview.net/forum?id=UaAD-Nu86WX) DiGress: Discrete Denoising Diffusion for Graph Generation, ICLR 2023.  
[[Jo et al. 23]](https://arxiv.org/abs/2302.03596) Graph Generation with Destiation-Driven Diffusion Mixture, arXiv preprint, 2023.  
