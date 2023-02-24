# Generative and Unsupervised Deep Learning @ KAIST

## Course Information
**Instructor:** Sung Ju Hwang (sjhwang82@kaist.ac.kr)  
**TAs:** Seul Lee (animecult@kaist.ac.kr) and Jaehyeong Jo (harryjo97@kaist.ac.kr)

**Office:** 
This is an on/offline hybrid course.
Building Nubmer 9, Room 9201 (Instructor) 2nd floor (TAs)  
Office hours: By appointment only.

### Grading Policy
* **Absolute Grading**
* Paper Presentation: 20%
* Attendance and Participation: 20%
* Assignments and Project: 60% 

## Tentative Schedule

| Dates | Topic | 
|---|:---|
|2/28| Course Introduction |
|3/2| Autoencoders and Variational Autoencoders (Lecture) |
|3/7| Transformers (Lecture) |
|3/9| Transformers (Lecture) |
|3/14| Self-Supervised Learning (Lecture) |
|3/16| Self-Supervised Learning (Presentation) | 
|3/21| Introduction to Deep Generative Models, GAN Basics (Lecture) | 
|3/23| Advanced GANs (Lecture) |
|3/28| Advanced GANs (Presentation) |
|3/30| Autoregressive and Flow-based Models (Lecture) | 
|4/4| Autoregressive and Flow-based Models (Presentation) |
|4/6| Diffusion Models (Lecture) |
|4/11| Diffusion Models (Lecture) |
|4/13| Diffusion Models (Presentation) | 
|4/18| **Mid-term Presentation**
|4/25| Language Models (Lecture)
|4/27| Large Language Models (Lecture) |
|5/2| Code Generation (Lecture) 
|5/4| Language Models and Code Generation (Presentation) |
|5/9| Text-to-Image Generation (Lecture) |
|5/11| Text-to-Image Generation (Presentation) | 
|5/16| Speech Synthesis (Lecture) | 
|5/23| Speech Synthesis (Presentation) |  
|5/25| Virtual Humans (Lecture)
|5/30| Virtual Humans (Presentation)
|6/1| Graph Representation Learning (Lecture) | 
|6/6| Graph Generation (Lecture) | 
|6/8| Graph Generation (Presentation) 
|6/13| **Final Presentation**

## Reading List

### Vision Transformers
[[Dosovitskiy et al. 21]](https://openreview.net/forum?id=YicbFdNTTy) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021.  
[[Touvron et al. 21]](http://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf) Training Data-efficient Image transformers & Distillation through Attention, ICML 2021.  
[[Liu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, ICCV 2021.  
[[Wu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf) CvT: Introducing Convolutions to Vision Transformers, ICCV 2021.  
[[Dai et al. 21]](https://proceedings.neurips.cc/paper/2021/file/20568692db622456cc42a2e853ca21f8-Paper.pdf) CoAtNet: Marrying Convolution and Attnetion for All Data Sizes, NeurIPS 2021.  
[[Yang et al. 21]](https://proceedings.neurips.cc/paper/2021/file/fc1a36821b02abbd2503fd949bfc9131-Paper.pdf) Focal Attention for Long-Range Interactions in Vision Transformers, NeurIPS 2021.  
[[El-Nouby et al. 21]](https://proceedings.neurips.cc/paper/2021/file/a655fbe4b8d7439994aa37ddad80de56-Paper.pdf) XCiT: Cross-Covariance Image Transformers, NeurIPS 2021.  
[[Li et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.pdf) MViTv2: Improved Multiscale Vision Transformers for Classification and Detection, CVPR 2022.  
[[Lee et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_MPViT_Multi-Path_Vision_Transformer_for_Dense_Prediction_CVPR_2022_paper.pdf) MPViT : Multi-Path Vision Transformer for Dense Prediction, CVPR 2022.  
[[Liu et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)A ConvNet for the 2020s, CVPR 2022.  


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
***
[[Liu et al. 22]](https://openreview.net/forum?id=4AZz9osqrar) Self-supervised Learning is More Robust to Dataset Imbalance, ICLR 2022.  
[[Bao et al. 22]](https://openreview.net/pdf?id=p-BhZSz59o4) BEiT: BERT Pre-Training of Image Transformers, ICLR 2022.  
[[He et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) Masked Autoencoders are Scalable Vision Learners, CVPR 2022.  
[[Liu et al. 22]](https://arxiv.org/pdf/2203.15508.pdf) Improving Contrastive Learning with Model Augmetnation, arXiv preprint, 2022.  
[[Touvron et al. 22]](https://arxiv.org/abs/2204.07118) DeIT III: Revenge of the VIT, arXiv preprint, 2022.

### Bayesian Deep Learning
[[Kingma and Welling 14]](https://arxiv.org/pdf/1312.6114.pdf) Auto-Encoding Variational Bayes, ICLR 2014.   
[[Kingma et al. 15]](https://arxiv.org/pdf/1506.02557.pdf) Variational Dropout and the Local Reparameterization Trick, NIPS 2015.   
[[Blundell et al. 15]](https://arxiv.org/pdf/1505.05424.pdf) Weight Uncertainty in Neural Networks, ICML 2015.   
[[Gal and Ghahramani 16]](http://proceedings.mlr.press/v48/gal16.pdf) Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, ICML 2016.   
[[Liu et al. 16]](https://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm.pdf) Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm, NIPS 2016.  
[[Mandt et al. 17]](https://www.jmlr.org/papers/volume18/17-214/17-214.pdf) Stochastic Gradient Descent as Approximate Bayesian Inference, JMLR 2017.  
[[Kendal and Gal 17]](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf) What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, ICML 2017.  
[[Gal et al. 17]](https://papers.nips.cc/paper/6949-concrete-dropout.pdf) Concrete Dropout, NIPS 2017.  
[[Gal et al. 17]](http://proceedings.mlr.press/v70/gal17a/gal17a.pdf) Deep Bayesian Active Learning with Image Data, ICML 2017.  
[[Teye et al. 18]](http://proceedings.mlr.press/v80/teye18a/teye18a.pdf) Bayesian Uncertainty Estimation for Batch Normalized Deep Networks, ICML 2018.  
[[Garnelo et al. 18]](http://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf) Conditional Neural Process, ICML 2018.  
[[Kim et al. 19]](http://https://arxiv.org/pdf/1901.05761.pdf) Attentive Neural Processes, ICLR 2019.  
[[Sun et al. 19]](https://arxiv.org/pdf/1903.05779.pdf) Functional Variational Bayesian Neural Networks, ICLR 2019.  
[[Louizos et al. 19]](http://papers.nips.cc/paper/9079-the-functional-neural-process.pdf) The Functional Neural Process, NeurIPS 2019.  
[[Zhang et al. 20]](https://openreview.net/forum?id=rkeS1RVtPS) Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning, ICLR 2020.  
[[Amersfoort et al. 20]](https://arxiv.org/pdf/2003.02037.pdf) Uncertainty Estimation Using a Single Deep Deterministic Neural Network, ICML 2020.  
[[Dusenberry et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/5657-Paper.pdf) Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors, ICML 2020.  
[[Wenzel et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/3581-Paper.pdf) How Good is the Bayes Posterior in Deep Neural Networks Really?, ICML 2020.  
[[Lee et al. 20]](https://arxiv.org/abs/2008.02956) Bootstrapping Neural Processes, NeurIPS 2020.  
[[Wilson et al. 20]](https://proceedings.neurips.cc/paper/2020/file/322f62469c5e3c7dc3e58f5a4d1ea399-Paper.pdf) Bayesian Deep Learning and a Probabilistic
Perspective of Generalization, NeurIPS 2020.  
[[Izmailov et al. 21]](http://proceedings.mlr.press/v139/izmailov21a/izmailov21a.pdf) What Are Bayesian Neural Network Posteriors Really Like?, ICML 2021.  
[[Daxberger et al. 21]](http://proceedings.mlr.press/v139/daxberger21a/daxberger21a.pdf) Bayesian Deep Learning via Subnetwork Inference, ICML 2021.  
***
[[Fortuin et al. 22]](https://openreview.net/forum?id=xkjqJYqRJy) Bayesian Neural Network Priors Revisited, ICLR 2022.  
[[Muller et al. 22]](https://openreview.net/pdf?id=KSugKcbNf9) Transformers Can Do Bayesian Inference, ICLR 2022.  
[[Nguyen and Grover 22]](https://proceedings.mlr.press/v162/nguyen22b/nguyen22b.pdf) Transformer Neural Processes, ICML 2022.  
[[Nazaret and Blei 22]](https://proceedings.mlr.press/v162/nazaret22a/nazaret22a.pdf) Variational Inference for Infinitely Deep Neural Networks, ICML 2022.  
[[Lotfi et al. 22]](https://proceedings.mlr.press/v162/lotfi22a/lotfi22a.pdf) Bayesian Model Selection, the Marginal Likelihood, and Generalization, ICML 2022.  
[[Alexos et al. 22]](https://proceedings.mlr.press/v162/alexos22a/alexos22a.pdf) Structured Stochastic Gradient MCMC, ICML 2022.  


### Deep Generative Models
#### VAEs, Autoregressive and Flow-Based Generative Models
[[Rezende and Mohamed 15]](http://proceedings.mlr.press/v37/rezende15.pdf) Variational Inference with Normalizing Flows, ICML 2015.   
[[Germain et al. 15]](http://proceedings.mlr.press/v37/germain15.pdf) MADE: Masked Autoencoder for Distribution Estimation, ICML 2015.  
[[Kingma et al. 16]](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow.pdf) Improved Variational Inference with Inverse Autoregressive Flow, NIPS 2016.  
[[Oord et al. 16]](http://proceedings.mlr.press/v48/oord16.pdf) Pixel Recurrent Neural Networks, ICML 2016.  
[[Dinh et al. 17]](https://openreview.net/pdf?id=HkpbnH9lx) Density Estimation Using Real NVP, ICLR 2017.  
[[Papamakarios et al. 17]](https://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation.pdf) Masked Autoregressive Flow for Density Estimation, NIPS 2017.  
[[Huang et al.18]](http://proceedings.mlr.press/v80/huang18d/huang18d.pdf) Neural Autoregressive Flows, ICML 2018.  
[[Kingma and Dhariwal 18]](http://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions.pdf) Glow: Generative Flow with Invertible 1x1 Convolutions, NeurIPS 2018.  
[[Ho et al. 19]](http://proceedings.mlr.press/v97/ho19a/ho19a.pdf) Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design, ICML 2019.    
[[Chen et al. 19]](https://papers.nips.cc/paper/9183-residual-flows-for-invertible-generative-modeling.pdf) Residual Flows for Invertible Generative Modeling, NeurIPS 2019.  
[[Tran et al. 19]](https://papers.nips.cc/paper/9612-discrete-flows-invertible-generative-models-of-discrete-data.pdf) Discrete Flows: Invertible Generative
Models of Discrete Data, NeurIPS 2019.  
[[Ping et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/647-Paper.pdf) WaveFlow: A Compact Flow-based Model for Raw Audio, ICML 2020.  
[[Vahdat and Kautz 20]](https://arxiv.org/pdf/2007.03898v1.pdf) NVAE: A Deep Hierarchical Variational Autoencoder, NeurIPS 2020.  
[[Ho et al. 20]](https://proceedings.neurips.cc//paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) Denoising Diffusion Probabilistic Models, NeurIPS 2020.  
[[Song et al. 21]](https://openreview.net/forum?id=PxTIG12RRHS) Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 2021.  
[[Kosiorek et al. 21]](http://proceedings.mlr.press/v139/kosiorek21a/kosiorek21a.pdf) NeRF-VAE: A Geometry Aware 3D Scene Generative Model, ICML 2021.  
***


#### Generative Adversarial Networks
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
***
[[Skorokhodov et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Skorokhodov_StyleGAN-V_A_Continuous_Video_Generator_With_the_Price_Image_Quality_CVPR_2022_paper.pdf) StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2, CVPR 2022.  
[[Lin et al. 22]](https://openreview.net/pdf?id=ufGMqIM0a4b) InfinityGAN: Towards Infinite-Pixel Image Synthesis, ICLR 2022.  
[[Lee et al. 22]](https://openreview.net/pdf?id=dwg5rXg1WS_) ViTGAN: Training GANs with Vision Transformers, ICLR 2022.  
[[Yu et al. 22]](https://openreview.net/forum?id=pfNyExj7z2) Vector-Quantized Image Modeling with Improved VQGAN, ICLR 2022.  
[[Franceschi et al. 22]](https://proceedings.mlr.press/v162/franceschi22a/franceschi22a.pdf) A Neural Tangent Kernel Perspective of GANs, ICML 2022.  

#### Diffusion Models
[[Song and Ermon 19]](https://proceedings.neurips.cc/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf) Generative Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2019.  
[[Song and Ermon 20]](https://papers.nips.cc/paper/2020/file/92c3b916311a5517d9290576e3ea37ad-Paper.pdf) Improved Techniques for Training Score-Based Generative Models, NeurIPS 2020.  
[[Ho et al. 20]](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) Denoising Diffusion Probabilistic Models, NeurIPS 2020.  
[[Song et al. 21]](https://openreview.net/forum?id=PxTIG12RRHS) Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 2021.  
[[Nichol and Dhariwal 21]](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) Improved Denoising Diffusion Probabilistic Models, ICML 2021.  
[[Vahdat et al. 21]](https://proceedings.neurips.cc/paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Paper.pdf) Score-based Generative Modeling in Latent Space, NeurIPS 2021.  
[[Dhariwal and Nichol 21]](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf) Diffusion Models Beat GANs on Image Synthesis, NeureIPS 2021.  
[[De Bortoli et al. 22]](https://proceedings.neurips.cc/paper/2021/file/940392f5f32a7ade1cc201767cf83e31-Paper.pdf) Diffusion Schrodinger Bridge with Application to Score-Based Generative Modeling, NeurIPS 2021.  
[[Ho and Salimans 22]](https://arxiv.org/pdf/2207.12598.pdf) Classifier-Free Diffusion Guidance, arXiv preprint, 2022.
***
[[Dockhorn et al. 22]](https://openreview.net/forum?id=CzceR82CYc) Score-Based Generative Modeling with Critically-Damped Langevin Diffusion, ICLR 2022.  
[[Salimans and Ho 22]](https://openreview.net/pdf?id=TIdIXIpzhoI) Progressive Distillation for Fast Sampling of Diffusion Models, ICLR 2022.  
[[Chen et al. 22]](https://openreview.net/forum?id=nioAdKCEdXB) Likelihood Training of Schrodinger Bridge using Forward-Backwrad SDEs Theory, ICLR 2022. 


### Deep Reinforcement Learning 
[[Mnih et al. 13]](https://arxiv.org/pdf/1312.5602.pdf) Playing Atari with Deep Reinforcement Learning, NIPS Deep Learning Workshop 2013.  
[[Silver et al. 14]](http://proceedings.mlr.press/v32/silver14.pdf) Deterministic Policy Gradient Algorithms, ICML 2014.      
[[Schulman et al. 15]](https://arxiv.org/pdf/1502.05477.pdf) Trust Region Policy Optimization, ICML 2015.  
[[Lillicrap et al. 16]](https://arxiv.org/pdf/1509.02971.pdf) Continuous Control with Deep Reinforcement Learning, ICLR 2016.    
[[Schaul et al. 16]](https://arxiv.org/pdf/1511.05952.pdf) Prioritized Experience Replay, ICLR 2016.  
[[Wang et al. 16]](http://proceedings.mlr.press/v48/wangf16.pdf) Dueling Network Architectures for Deep Reinforcement Learning, ICML 2016.    
[[Mnih et al. 16]](http://proceedings.mlr.press/v48/mniha16.pdf) Asynchronous Methods for Deep Reinforcement Learning, ICML 2016.  
[[Schulman et al. 17]](https://arxiv.org/pdf/1707.06347.pdf) Proximal Policy Optimization Algorithms, arXiv preprint, 2017.  
[[Nachum et al. 18]](https://papers.nips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning.pdf) Data-Efficient Hierarchical Reinforcement Learning, NeurIPS 2018.  
[[Ha et al. 18]](https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution.pdf) Recurrent World Models Facilitate Policy Evolution, NeurIPS 2018.  
[[Burda et al. 19]](https://openreview.net/forum?id=rJNwDjAqYX) Large-Scale Study of Curiosity-Driven Learning, ICLR 2019.  
[[Vinyals et al. 19]](https://www.nature.com/articles/s41586-019-1724-z) Grandmaster level in StarCraft II using multi-agent reinforcement learning, Nature, 2019.  
[[Bellemare et al. 19]](https://papers.nips.cc/paper/8687-a-geometric-perspective-on-optimal-representations-for-reinforcement-learning.pdf) A Geometric Perspective on Optimal Representations for Reinforcement Learning, NeurIPS 2019.  
[[Janner et al. 19]](http://papers.nips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization.pdf) When to Trust Your Model: Model-Based Policy Optimization, NeurIPS 2019.  
[[Fellows et al. 19]](https://papers.nips.cc/paper/8934-virel-a-variational-inference-framework-for-reinforcement-learning.pdf) VIREL: A Variational Inference Framework for Reinforcement Learning, NeurIPS 2019.  
[[Kumar et al. 19]](https://proceedings.neurips.cc/paper/2019/file/c2073ffa77b5357a498057413bb09d3a-Paper.pdf) Stabilizing Off-Policy Q-Learning via Bootstrapping
Error Reduction, NeurIPS 2019.  
[[Kaiser et al. 20]](https://openreview.net/pdf?id=S1xCPJHtDB) Model Based Reinforcement Learning for Atari, ICLR 2020.  
[[Agarwal et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/5394-Paper.pdf) An Optimistic Perspective on Offline Reinforcement Learning, ICML 2020.  
[[Lee et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/3705-Paper.pdf) Batch Reinforcement Learning with Hyperparameter Gradients, ICML 2020.  
[[Kumar et al. 20]](https://proceedings.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf) Conservative Q-Learning for Offline Reinforcement Learning, ICML 2020.  
[[Yarats et al. 21]](https://openreview.net/pdf?id=GY6-6sTvGaf) Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels, ICLR 2021.  
[[Chen et al. 21]](https://proceedings.neurips.cc/paper/2021/file/7f489f642a0ddb10272b5c31057f0663-Paper.pdf) Decision Transformer: Reinforcement Learning via Sequence Modeling, NeurIPS 2021.  
***
[[Mai et al. 22]](https://openreview.net/pdf?id=vrW3tvDfOJQ) Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation, ICLR 2022.  
[[Furuta et al. 22]](https://openreview.net/forum?id=CAjxVodl_v) Generalized Decision Transformer for Offline Hindsight Information Matching, ICLR 2022.  
[[Oh et al. 22]](https://openreview.net/forum?id=WuEiafqdy9H) Model-augmented Prioritized Experience Replay, ICLR 2022.  
[[Rengarajan et al. 22]](https://openreview.net/pdf?id=YJ1WzgMVsMt) Reinforcement Learning with Sparse Rewards Using Guidance from Offline Demonstration, ICLR 2022.  
[[Patil et al. 22]](https://proceedings.mlr.press/v162/patil22a/patil22a.pdf) Align-RUDDER: Learning from Few Demonstrations by Reward Redistribution, ICML 2022.  
[[Goyal et al. 22]](https://proceedings.mlr.press/v162/goyal22a/goyal22a.pdf) Retrieval Augmented Reinforcement Learning, ICML 2022.  
[[Reed et al. 22]](https://arxiv.org/pdf/2205.06175.pdf) A Generalist Agent, arXiv preprint, 2022.  

### Memory and Computation-Efficient Deep Learning
[[Han et al. 15]](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf) Learning both Weights and Connections for Efficient Neural Networks, NIPS 2015.  
[[Wen et al. 16]](https://papers.nips.cc/paper/6504-learning-structured-sparsity-in-deep-neural-networks.pdf) Learning Structured Sparsity in Deep Neural Networks, NIPS 2016  
[[Han et al. 16]](https://arxiv.org/pdf/1510.00149.pdf) Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding, ICLR 2016  
[[Molchanov et al. 17]](https://arxiv.org/pdf/1701.05369.pdf) Variational Dropout Sparsifies Deep Neural Networks, ICML 2017  
[[Luizos et al. 17]](https://papers.nips.cc/paper/6921-bayesian-compression-for-deep-learning.pdf) Bayesian Compression for Deep Learning, NIPS 2017.  
[[Luizos et al. 18]](https://openreview.net/pdf?id=H1Y8hhg0b) Learning Sparse Neural Networks Through L0 Regularization, ICLR 2018.    
[[Howard et al. 18]](https://arxiv.org/pdf/1704.04861.pdf) MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications, CVPR 2018.    
[[Frankle and Carbin 19]](https://https://openreview.net/pdf?id=rJl-b3RcF7) The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, ICLR 2019.     
[[Lee et al. 19]](https://openreview.net/pdf?id=B1VZqjAcYX) SNIP: Single-Shot Network Pruning Based On Connection Sensitivity, ICLR 2019.  
[[Liu et al. 19]](https://openreview.net/pdf?id=rJlnB3C5Ym) Rethinking the Value of Network Pruning, ICLR 2019.  
[[Jung et al. 19]](https://arxiv.org/abs/1808.05779) Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss, CVPR 2019.  
[[Morcos et al. 19]](https://papers.nips.cc/paper/8739-one-ticket-to-win-them-all-generalizing-lottery-ticket-initializations-across-datasets-and-optimizers.pdf) One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers, NeurIPS 2019.  
[[Renda et al. 20]](https://openreview.net/pdf?id=S1gSj0NKvB) Comparing Rewinding and Fine-tuning in Neural Network Pruning, ICLR 2020.  
[[Frankle et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/5787-Paper.pdf) Linear Mode Connectivity and the Lottery Ticket Hypothesis, ICML 2020.  
[[Tanaka et al. 20]](https://proceedings.neurips.cc/paper/2020/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf) Pruning Neural Networks without Any Data by Iteratively Conserving Synaptic Flow, NeurIPS 2020.  
[[van Baalen et al. 20]](https://proceedings.neurips.cc/paper/2020/file/3f13cf4ddf6fc50c0d39a1d5aeb57dd8-Paper.pdf) Bayesian Bits: Unifying Quantization and Pruning, NeurIPS 2020.  
[[de Jorge et al. 21]](https://openreview.net/pdf?id=9GsFOUyUPi) Progressive Skeletonization: Trimming more fat from a network at initialization, ICLR 2021.  
[[Stock et al. 21]](https://openreview.net/pdf?id=dV19Yyi1fS3) Training with Quantization Noise for Extreme Model Compression, ICLR 2021.  
[[Lee et al. 21]](https://arxiv.org/abs/1911.12990) Semi-Relaxed Quantization with DropBits: Training Low-Bit Neural Networks via Bit-wise Regularization, ICCV 2021.  
***


### Meta Learning
[[Santoro et al. 16]](http://proceedings.mlr.press/v48/santoro16.pdf) Meta-Learning with Memory-Augmented Neural Networks, ICML 2016  
[[Vinyals et al. 16]](https://arxiv.org/pdf/1606.04080.pdf) Matching Networks for One Shot Learning, NIPS 2016    
[[Edwards and Storkey 17]](https://arxiv.org/pdf/1606.02185.pdf) Towards a Neural Statistician, ICLR 2017  
[[Finn et al. 17]](https://arxiv.org/pdf/1703.03400.pdf) Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, ICML 2017  
[[Snell et al. 17]](https://arxiv.org/pdf/1703.05175.pdf) Prototypical Networks for Few-shot Learning, NIPS 2017.  
[[Nichol et al. 18]](https://arxiv.org/pdf/1803.02999.pdf) On First-Order Meta-learning Algorithms, arXiv preprint, 2018.  
[[Lee and Choi 18]](https://arxiv.org/abs/1801.05558) Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace, ICML 2018.  
[[Liu et al. 19]](https://openreview.net/pdf?id=SyVuRiC5K7) Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning, ICLR 2019.    
[[Gordon et al. 19]](https://openreview.net/pdf?id=HkxStoC5F7) Meta-Learning Probabilistic Inference for Prediction, ICLR 2019.  
[[Ravi and Beatson 19]](https://openreview.net/pdf?id=rkgpy3C5tX) Amortized Bayesian Meta-Learning, ICLR 2019.    
[[Rakelly et al. 19]](http://proceedings.mlr.press/v97/rakelly19a/rakelly19a.pdf) Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables, ICML 2019.  
[[Shu et al. 19]](https://papers.nips.cc/paper/8467-meta-weight-net-learning-an-explicit-mapping-for-sample-weighting.pdf) Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting, NeurIPS 2019.  
[[Finn et al. 19]](http://proceedings.mlr.press/v97/finn19a/finn19a.pdf) Online Meta-Learning, ICML 2019.  
[[Lee et al. 20]](https://openreview.net/pdf?id=rkeZIJBYvr) Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks, ICLR 2020.  
[[Yin et al. 20]](https://openreview.net/forum?id=BklEFpEYwS) Meta-Learning without Memorization, ICLR 2020.  
[[Raghu et al. 20]](https://openreview.net/pdf?id=rkgMkCEtPB) Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML, ICLR 2020.  
[[Iakovleva et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/4070-Paper.pdf) Meta-Learning with Shared Amortized Variational Inference, ICML 2020.   
[[Bronskill et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/2696-Paper.pdf) TaskNorm: Rethinking Batch Normalization for Meta-Learning, ICML 2020.   
[[Rajendran et al. 20]](https://proceedings.neurips.cc//paper/2020/file/3e5190eeb51ebe6c5bbc54ee8950c548-Paper.pdf) Meta-Learning Requires Meta-Augmentation, NeurIPS 2020.  
[[Lee et al. 21]](https://openreview.net/forum?id=wS0UFjsNYjn) Meta-GMVAE: Mixture of Gaussian VAE for Unsupervised Meta-Learning, ICLR 2021.  
[[Shin et al. 21]](http://proceedings.mlr.press/v139/shin21a/shin21a.pdf) Large-Scale Meta-Learning with Continual Trajectory Shifting, ICML 2021.  
[[Acar et al. 21]](http://proceedings.mlr.press/v139/acar21b/acar21b.pdf) Memory Efficient Online Meta Learning, ICML 2021.  
***
[[Lee et al. 22]](https://openreview.net/forum?id=01AMRlen9wJ) Online Hyperparameter Meta-Learning with Hypergradient Distillation, ICLR 2022.  
[[Flennerhag et al. 22]](https://openreview.net/forum?id=b-ny3x071E5) Boostrapped Meta-Learning, ICLR 2022.  
[[Yao et al. 22]](https://openreview.net/forum?id=ajXWF7bVR8d) Meta-Learning with Fewer Tasks through Task Interpolation, ICLR 2022.  
[[Guan and Lu 22]](https://openreview.net/pdf?id=A3HHaEdqAJL) Task Relatedness-Based Generalization Bounds for Meta Learning, ICLR 2022.  

### Continual Learning
[[Rusu et al. 16]](https://arxiv.org/pdf/1606.04671.pdf) Progressive Neural Networks, arXiv preprint, 2016  
[[Kirkpatrick et al. 17]](https://arxiv.org/pdf/1612.00796.pdf) Overcoming catastrophic forgetting in neural networks, PNAS 2017  
[[Lee et al. 17]](https://papers.nips.cc/paper/7051-overcoming-catastrophic-forgetting-by-incremental-moment-matching.pdf) Overcoming Catastrophic Forgetting by Incremental Moment Matching, NIPS 2017  
[[Shin et al. 17]](https://papers.nips.cc/paper/6892-continual-learning-with-deep-generative-replay.pdf) Continual Learning with Deep Generative Replay, NIPS 2017.  
[[Lopez-Paz and Ranzato 17]](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf) Gradient Episodic Memory for Continual Learning, NIPS 2017.  
[[Yoon et al. 18]](https://openreview.net/pdf?id=Sk7KsfW0-) Lifelong Learning with Dynamically Expandable Networks, ICLR 2018.  
[[Nguyen et al. 18]](https://arxiv.org/pdf/1710.10628.pdf) Variational Continual Learning, ICLR 2018.  
[[Schwarz et al. 18]](http://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) Progress & Compress: A Scalable Framework for Continual Learning, ICML 2018.   
[[Chaudhry et al. 19]](https://openreview.net/pdf?id=Hkf2_sC5FX) Efficient Lifelong Learning with A-GEM, ICLR 2019.  
[[Rao et al. 19]](https://papers.nips.cc/paper/8981-continual-unsupervised-representation-learning.pdf) Continual Unsupervised Representation Learning, NeurIPS 2019.  
[[Rolnick et al. 19]](https://papers.nips.cc/paper/8327-experience-replay-for-continual-learning.pdf) Experience Replay for Continual Learning, NeurIPS 2019.  
[[Jerfel et al. 20]](https://papers.nips.cc/paper/9112-reconciling-meta-learning-and-continual-learning-with-online-mixtures-of-tasks.pdf) Reconciling Meta-Learning and Continual Learning with Online Mixtures of Tasks, NeurIPS 2019.  
[[Yoon et al. 20]](https://openreview.net/forum?id=r1gdj2EKPB) Scalable and Order-robust Continual Learning with Additive Parameter Decomposition, ICLR 2020.  
[[Remasesh et al. 20]](https://arxiv.org/pdf/2007.07400.pdf) Anatomy of Catastrophic Forgetting: Hidden Representations and Task Semantics, Continual Learning Workshop, ICML 2020.  
[[Borsos et al. 20]](https://proceedings.neurips.cc/paper/2020/file/aa2a77371374094fe9e0bc1de3f94ed9-Paper.pdf) Coresets via Bilevel Optimization for Continual
Learning and Streaming, NeurIPS 2020.  
[[Mirzadeh et al. 20]](https://proceedings.neurips.cc/paper/2020/file/518a38cc9a0173d0b2dc088166981cf8-Paper.pdf) Understanding the Role of Training Regimes
in Continual Learning, NeurIPS 2020.  
[[Saha et al. 21]](https://openreview.net/pdf?id=3AOj0RCNC2) Gradient Projection Memory for Continual Learning, ICLR 2021.  
[[Veinat et al. 21]](https://openreview.net/pdf?id=EKV158tSfwv) Efficient Continual Learning with Modular Networks and Task-Driven Priors, ICLR 2021.  
***
[[Madaan et al. 22]](https://openreview.net/forum?id=9Hrka5PA7LW) Representational Continuity for Unsupervised Continual Learning, ICLR 2022.    
[[Yoon et al. 22]](https://openreview.net/forum?id=f9D-5WNG4Nv) Online Coreset Selection for Rehearsal-based Continual Learning, ICLR 2022.  
[[Lin et al. 22]](https://openreview.net/pdf?id=iEvAf8i6JjO) TRGP: Trust Region Gradient Projection for Continual Learning, ICLR 2022.  
[[Wang et al. 22]](https://proceedings.mlr.press/v162/wang22v/wang22v.pdf) Improving Task-free Continual Learning by Distributionally Robust Memory Evolution, ICML 2022.  
[[Kang et al. 22]](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) Forget-free Continual Learning with Winning Subnetworks, ICML 2022.  





### Interpretable Deep Learning
[[Ribeiro et al. 16]](https://arxiv.org/pdf/1602.04938.pdf) "Why Should I Trust You?" Explaining the Predictions of Any Classifier, KDD 2016  
[[Kim et al. 16]](https://people.csail.mit.edu/beenkim/papers/KIM2016NIPS_MMD.pdf) Examples are not Enough, Learn to Criticize! Criticism for Interpretability, NIPS 2016  
[[Choi et al. 16]](https://papers.nips.cc/paper/6321-retain-an-interpretable-predictive-model-for-healthcare-using-reverse-time-attention-mechanism.pdf) RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism, NIPS 2016  
[[Koh et al. 17]](https://arxiv.org/pdf/1703.04730.pdf) Understanding Black-box Predictions via Influence Functions, ICML 2017  
[[Bau et al. 17]](https://arxiv.org/pdf/1704.05796.pdf) Network Dissection: Quantifying Interpretability of Deep Visual Representations, CVPR 2017  
[[Selvaraju et al. 17]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf) Grad-CAM: Visual Explanation from Deep Networks via Gradient-based Localization, ICCV 2017.  
[[Kim et al. 18]](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf) Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV), ICML 2018.  
[[Heo et al. 18]](http://papers.nips.cc/paper/7370-uncertainty-aware-attention-for-reliable-interpretation-and-prediction.pdf) Uncertainty-Aware Attention for Reliable Interpretation and Prediction, NeurIPS 2018.   
[[Bau et al. 19]](https://openreview.net/pdf?id=Hyg_X2C5FX) GAN Dissection: Visualizing and Understanding Generative Adversarial Networks, ICLR 2019.   
[[Ghorbani et al. 19]](https://papers.nips.cc/paper/9126-towards-automatic-concept-based-explanations.pdf) Towards Automatic Concept-based Explanations, NeurIPS 2019.  
[[Coenen et al. 19]](https://papers.nips.cc/paper/9065-visualizing-and-measuring-the-geometry-of-bert.pdf) Visualizing and Measuring the Geometry of BERT, NeurIPS 2019.  
[[Heo et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/579-Paper.pdf) Cost-Effective Interactive Attention Learning with Neural Attention Processes, ICML 2020.  
[[Agarwal et al. 20]](https://arxiv.org/pdf/2004.13912v1.pdf) Neural Additive Models: Interpretable Machine Learning with Neural Nets, arXiv preprint, 2020.  
***

### Reliable Deep Learning
[[Guo et al. 17]](https://arxiv.org/pdf/1706.04599.pdf) On Calibration of Modern Neural Networks, ICML 2017.   
[[Lakshminarayanan et al. 17]](http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf) Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, NIPS 2017.  
[[Liang et al. 18]](https://openreview.net/pdf?id=H1VGkIxRZ) Enhancing the Reliability of Out-of-distrubition Image Detection in Neural Networks, ICLR 2018.  
[[Lee et al. 18]](https://openreview.net/pdf?id=ryiAv2xAZ) Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples, ICLR 2018.  
[[Kuleshov et al. 18]](https://arxiv.org/pdf/1807.00263.pdf) Accurate Uncertainties for Deep Learning Using Calibrated Regression, ICML 2018.    
[[Jiang et al. 18]](https://papers.nips.cc/paper/7798-to-trust-or-not-to-trust-a-classifier.pdf) To Trust Or Not To Trust A Classifier, NeurIPS 2018.  
[[Madras et al. 18]](https://papers.nips.cc/paper/7853-predict-responsibly-improving-fairness-and-accuracy-by-learning-to-defer.pdf) Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer, NeurIPS 2018.  
[[Maddox et al. 19]](https://papers.nips.cc/paper/9472-a-simple-baseline-for-bayesian-uncertainty-in-deep-learning.pdf) A Simple Baseline for Bayesian Uncertainty in Deep Learning, NeurIPS 2019.  
[[Kull et al. 19]](https://papers.nips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf) Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration, NeurIPS 2019.  
[[Thulasidasan et al. 19]](https://papers.nips.cc/paper/9540-on-mixup-training-improved-calibration-and-predictive-uncertainty-for-deep-neural-networks.pdf) On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks, NeurIPS 2019.  
[[Ovadia et al. 19]](https://papers.nips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf) Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift, NeurIPS 2019.  
[[Hendrycks et al. 20]](https://openreview.net/pdf?id=S1gmrxHFvB) AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty, ICLR 2020.  
[[Filos et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/2969-Paper.pdf) Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?, ICML 2020.  
***


### Robust Deep Learning
[[Szegedy et al. 14]](https://arxiv.org/pdf/1312.6199.pdf) Intriguing Properties of Neural Networks, ICLR 2014.    
[[Goodfellow et al. 15]](https://arxiv.org/pdf/1412.6572.pdf) Explaining and Harnessing Adversarial Examples, ICLR 2015.    
[[Kurakin et al. 17]](https://openreview.net/pdf?id=BJm4T4Kgx) Adversarial Machine Learning at Scale, ICLR 2017.    
[[Madry et al. 18]](https://openreview.net/pdf?id=rJzIBfZAb) Toward Deep Learning Models Resistant to Adversarial Attacks, ICLR 2018.    
[[Eykholt et al. 18]](https://arxiv.org/pdf/1707.08945.pdf) Robust Physical-World Attacks on Deep Learning Visual Classification.  
[[Athalye et al. 18]](https://arxiv.org/pdf/1802.00420.pdf) Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples, ICML 2018.  
[[Zhang et al. 19]](http://proceedings.mlr.press/v97/zhang19p/zhang19p.pdf) Theoretically Principled Trade-off between Robustness and Accuracy, ICML 2019.  
[[Carmon et al. 19]](https://papers.nips.cc/paper/9298-unlabeled-data-improves-adversarial-robustness.pdf) Unlabeled Data Improves Adversarial Robustness, NeurIPS 2019.  
[[Ilyas et al. 19]](https://papers.nips.cc/paper/8307-adversarial-examples-are-not-bugs-they-are-features) Adversarial Examples are not Bugs, They Are Features, NeurIPS 2019.  
[[Li et al. 19]](https://papers.nips.cc/paper/9143-certified-adversarial-robustness-with-additive-noise.pdf) Certified Adversarial Robustness with Additive Noise, NeurIPS 2019.  
[[Tramèr and Boneh 19]](https://papers.nips.cc/paper/8821-adversarial-training-and-robustness-for-multiple-perturbations.pdf) Adversarial Training and Robustness for Multiple Perturbations, NeurIPS 2019.  
[[Shafahi et al. 19]](https://papers.nips.cc/paper/8597-adversarial-training-for-free.pdf) Adversarial Training for Free!, NeurIPS 2019.  
[[Wong et al. 20]](https://openreview.net/pdf?id=BJx040EFvH) Fast is Better Than Free: Revisiting Adversarial Training, ICLR 2020.  
[[Madaan et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/770-Paper.pdf) Adversarial Neural Pruning with Latent Vulnerability Suppression, ICML 2020.  
[[Croce and Hein 20]](http://proceedings.mlr.press/v119/croce20b/croce20b.pdf) Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks, ICML 2020.  
[[Maini et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/3863-Paper.pdf) Adversarial Robustness Against the Union of Multiple Perturbation Models, ICML 2020.  
[[Kim et al. 20]](https://proceedings.neurips.cc/paper/2020/file/1f1baa5b8edac74eb4eaa329f14a0361-Paper.pdf) Adversarial Self-Supervised Contrastive Learning, NeurIPS 2020.  
[[Wu et al. 20]](https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf) Adversarial Weight Perturbation Helps Robust Generalization, NeurIPS 2020.  
[[Laidlaw et al. 21]](https://openreview.net/pdf?id=dFwBosAcJkN) Perceptual Adversarial Robustness: Defense Against Unseen Threat Models, ICLR 2021.  
[[Pang et al. 21]](https://openreview.net/pdf?id=Xb8xvrtB8Ce) Bag of Tricks for Adversarial Training, ICLR 2021.  
[[Madaan et al. 21]](http://proceedings.mlr.press/v139/madaan21a/madaan21a.pdf) Learning to Generate Noise for Multi-Attack Robustness, ICML 2021.  
***
[[Mladenovic et al. 22]](https://openreview.net/pdf?id=bYGSzbCM_i) Online Adversarial Attacks, ICLR 2022.   
[[Zhang et al. 22]](https://openreview.net/pdf?id=W9G_ImpHlQd) How to Robustify Black-Box ML Models? A Zeroth-Order Optimization Perspective, ICLR 2022.  
[[Carlini and Terzis 22]](https://openreview.net/pdf?id=iC4UHbQ01Mp) Poisoning and Backdooring Contrastive Learning, ICLR 2022.   
[[Croce et al. 22]](https://proceedings.mlr.press/v162/croce22a/croce22a.pdf) Evaluating the Adversarial Robustness of Adaptive Test-time Defenses, ICML 2022.  
[[Zhou et al. 22]](https://proceedings.mlr.press/v162/zhou22m/zhou22m.pdf) Understanding the Robustness in Vision Transformers, ICML 2022.  


### Graph Neural Networks
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
***
[[Guo et al. 22]](https://openreview.net/pdf?id=l4IHywGq6a) Data-Efficient Graph Grammar Learning for Molecular Generation, ICLR 2022.  
[[Geerts et al. 22]](https://openreview.net/pdf?id=wIzUeM3TAU) Expressiveness and Approximation Properties of Graph Neural Networks, ICLR 2022.  
[[Bevilacqua et al. 22]](https://openreview.net/pdf?id=dFbKQaRk15w) Equivariant Subgraph Aggregation Networks, ICLR 2022.  
[[Jo et al. 22]](https://proceedings.mlr.press/v162/jo22a/jo22a.pdf) Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations, ICML 2022.  
[[Hoogeboom et al. 22]](https://proceedings.mlr.press/v162/hoogeboom22a/hoogeboom22a.pdf) Equivariant Diffusion for Molecule Generation in 3D, ICML 2022.  


### Federated Learning
[[Konečný et al. 16]](https://arxiv.org/pdf/1610.02527.pdf) Federated Optimization: Distributed Machine Learning for On-Device Intelligence, arXiv Preprint, 2016.  
[[Konečný et al. 16]](https://arxiv.org/abs/1610.05492) Federated Learning: Strategies for Improving Communication Efficiency, NIPS Workshop on Private Multi-Party Machine Learning 2016.  
[[McMahan et al. 17]](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) Communication-Efficient Learning of Deep Networks from Decentralized Data, AISTATS 2017.   
[[Smith et al. 17]](https://papers.nips.cc/paper/7029-federated-multi-task-learning.pdf) Federated Multi-Task Learning, NIPS 2017.  
[[Li et al. 20]](https://proceedings.mlsys.org/static/paper_files/mlsys/2020/176-Paper.pdf) Federated Optimization in Heterogeneous Networks, MLSys 2020.   
[[Yurochkin et al. 19]](http://proceedings.mlr.press/v97/yurochkin19a/yurochkin19a.pdf) Bayesian Nonparametric Federated Learning of Neural Networks, ICML 2019.  
[[Bonawitz et al. 19]](https://arxiv.org/pdf/1902.01046.pdf) Towards Federated Learning at Scale: System Design, MLSys 2019.  
[[Wang et al. 20]](https://openreview.net/forum?id=BkluqlSFDS) Federated Learning with Matched Averaging, ICLR 2020.  
[[Li et al. 20]](https://openreview.net/pdf?id=HJxNAnVtDS) On the Convergence of FedAvg on Non-IID data, ICLR 2020.  
[[Karimireddy et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/788-Paper.pdf) SCAFFOLD: Stochastic Controlled Averaging for Federated Learning, ICML 2020.  
[[Hamer et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/5967-Paper.pdf) FedBoost: Communication-Efficient Algorithms for Federated Learning, ICML 2020.  
[[Rothchild et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/5927-Paper.pdf) FetchSGD: Communication-Efficient Federated Learning with Sketching, ICML 2020.   
[[Fallah et al. 21]](https://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf) Personalized Federated Learning with Theoretical
Guarantees: A Model-Agnostic Meta-Learning Approach, NeurIPS 2020.  
[[Reddi et al. 21]](https://openreview.net/pdf?id=LkFG3lB13U5) Adaptive Federated Optimization, ICLR 2021.  
[[Jeong et al. 21]](https://openreview.net/pdf?id=ce6CFXBh30h) Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning, ICLR 2021.  
[[Yoon et al. 21]](https://openreview.net/forum?id=Svfh1_hYEtF) Federated Continual Learning with Weighted Inter-client Transfer, ICML 2021.  
[[Li et al. 21]](http://proceedings.mlr.press/v139/li21h/li21h.pdf) Ditto: Fair and Robust Federated Learning Through Personalization, ICML 2021.  
***


### Neural Architecture Search 
[[Zoph and Le 17]](https://arxiv.org/abs/1611.01578) Neural Architecture Search with Reinforcement Learning, ICLR 2017.  
[[Baker et al. 17]](https://openreview.net/pdf?id=S1c2cvqee) Designing Neural Network Architectures using Reinforcement Learning, ICLR 2017.  
[[Real et al. 17]](http://proceedings.mlr.press/v70/real17a/real17a.pdf) Large-Scale Evolution of Image Classifiers, ICML 2017.  
[[Liu et al. 18]](https://openreview.net/pdf?id=BJQRKzbA-) Hierarchical Representations for Efficient Architecture Search, ICLR 2018.  
[[Pham et al. 18]](http://proceedings.mlr.press/v80/pham18a.html) Efficient Neural Architecture Search via Parameters Sharing, ICML 2018.  
[[Luo et al. 18]](https://papers.nips.cc/paper/8007-neural-architecture-optimization.pdf) Neural Architecture Optimization, NeurIPS 2018.  
[[Liu et al. 19]](https://openreview.net/pdf?id=S1eYHoC5FX) DARTS: Differentiable Architecture Search, ICLR 2019.  
[[Tan et al. 19]](https://arxiv.org/abs/1807.11626) MnasNet: Platform-Aware Neural Architecture Search for Mobile, CVPR 2019.  
[[Cai et al. 19]](https://openreview.net/pdf?id=HylVB3AqYm) ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware, ICLR 2019.  
[[Zhou et al. 19]](http://proceedings.mlr.press/v97/zhou19e/zhou19e.pdf) BayesNAS: A Bayesian Approach for Neural Architecture Search, ICML 2019.  
[[Tan and Le 19]](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf) EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019.  
[[Guo et al. 19]](https://papers.nips.cc/paper/8362-nat-neural-architecture-transformer-for-accurate-and-compact-architectures.pdf) NAT: Neural Architecture Transformer for Accurate and Compact Architectures, NeurIPS 2019.  
[[Chen et al. 19]](https://papers.nips.cc/paper/8890-detnas-backbone-search-for-object-detection.pdf) DetNAS: Backbone Search for Object Detection, NeurIPS 2019.  
[[Dong and Yang 20]](https://openreview.net/forum?id=HJxyZkBKDr) NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search, ICLR 2020.  
[[Zela et al. 20]](https://openreview.net/pdf?id=H1gDNyrKDS) Understanding and Robustifying Differentiable Architecture Search, ICLR 2020.  
[[Cai et al. 20]](https://openreview.net/pdf?id=HylxE1HKwS) Once-for-All: Train One Network and Specialize it for Efficient Deployment, ICLR 2020.  
[[Such et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/4522-Paper.pdf) Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data, ICML 2020.  
[[Liu et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490766.pdf) Are Labels Necessary for Neural Architecture Search?, ECCV 2020.  
[[Dudziak et al. 20]](https://proceedings.neurips.cc/paper/2020/file/768e78024aa8fdb9b8fe87be86f64745-Paper.pdf) BRP-NAS: Prediction-based NAS using GCNs, NeurIPS 2020.  
[[Li et al. 20]](https://proceedings.icml.cc/static/paper_files/icml/2020/439-Paper.pdf) Neural Architecture Search in A Proxy Validation Loss Landscape, ICML 2020.  
[[Lee et al. 21]](https://openreview.net/forum?id=rkQuFUmUOg3) Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021.   
[[Mellor et al. 21]](https://arxiv.org/pdf/2006.04647.pdf) Neural Architecture Search without Training, ICML 2021.  
***

### Large Language Models
[[Shoeybi et al. 19]](https://arxiv.org/abs/1909.08053) Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism, arXiv preprint, 2019.  
[[Raffel et al. 20]](https://jmlr.org/papers/volume21/20-074/20-074.pdf) Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, JMLR 2020.  
[[Gururangan et al. 20]](https://aclanthology.org/2020.acl-main.740.pdf) Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks, ACL 2020.  
[[Brown et al. 20]](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) Language Models are Few-shot Learners, NeurIPS 2020.  
[[Rae et al. 21]](https://arxiv.org/abs/2112.11446) Scaling Language Models: Methods, Analysis & Insights from Training Gopher, arXiv preprint, 2021.  
***
[[Thoppilan et al. 22]](https://arxiv.org/pdf/2201.11903.pdf) LaMDA: Language Models for Dialog Applications, arXiv preprint, 2022.   
[[Wei et al. 22]](https://openreview.net/pdf?id=gEZrGCozdqR) Finetuned Langauge Models Are Zero-Shot Learners, ICLR 2022.  
[[Wang et al. 22]](https://openreview.net/forum?id=pMQwKL1yctf) Language Modeling via Stochastic Processes, ICLR 2022.  
[[Alayrac et al. 22]](https://arxiv.org/abs/2204.14198) Flamingo: a Visual Language Model for Few-Shot Learning, arXiv preprint, 2022.  
[[Chowdhery et al. 22]](https://arxiv.org/abs/2204.02311) PaLM: Scaling Langauge Modeling with Pathways, arXiv preprint, 2022.  
[[Wei et al. 22]](https://arxiv.org/pdf/2201.11903.pdf) Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, NeurIPS 2022.  

### Multimodal Generative Models
[[Li et al. 19]](https://proceedings.neurips.cc/paper/2019/file/1d72310edc006dadf2190caad5802983-Paper.pdf) Controllable Text-to-Image Generation, NeurIPS 2019.  
[[Ramesh et al. 21]](http://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf) Zero-Shot Text-to-Image Generation, ICML 2021.   
[[Radford et al. 21]](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) Learning Transferable Visual Models From Natural Language Supervision, ICML 2021.  
[[Ding et al. 21]](https://proceedings.neurips.cc/paper/2021/file/a4d92e2cd541fca87e4620aba658316d-Paper.pdf) CogView: Mastering Text-to-Image Generation via Transformers, NeurIPS 2021.  
[[Zou et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Towards_Language-Free_Training_for_Text-to-Image_Generation_CVPR_2022_paper.pdf) Towards Language-Free Training for Text-to-Image Generation, CVPR 2022.  
***
[[Rombach et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) High-Resolution Image Synthesis with Latent Diffusion Models, CVPR 2022.  
[[Nichol et al. 22]](https://proceedings.mlr.press/v162/nichol22a/nichol22a.pdf) GLIDE: Towards Photorealistic Image Generation and Editing with
Text-Guided Diffusion Models, ICML 2022.  
[[Saharia et al. 22]](https://arxiv.org/abs/2205.11487) Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding, arXiv preprint, 2022.  
[[Yu et al. 22]](https://arxiv.org/abs/2206.10789) Scaling Autoregressive Models for Content-Rich Text-to-Image Generation, arXiv preprint, 2022.  


<!--
## Deep Learning for Time-series Analysis
[[Che et al. 18]](http://proceedings.mlr.press/v80/che18a/che18a.pdf) Hierarchical Deep Generative Models for Multi-Rate Multivariate Time Series, ICML 2018  
[[Campos et al. 18]](https://openreview.net/pdf?id=HkwVAXyCW) Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks, ICLR 2018  
[[Tatbul et al. 18]](https://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf) Precision and Recall for Time Series, NeurIPS 2018  
[[Rangapuram et al. 18]](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf) Deep State Space Models for Time Series Forecasting, NeurIPS 2018  
## Deep Learning for Computer Vision
[[Ren et al. 15]](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf_) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, NIPS 2015  
[[Liu et al. 16]](https://arxiv.org/pdf/1512.02325.pdf) SSD: Single Shot MultiBox Detector, ECCV 2016  
[[Long et al. 15]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) Fully Convolutional Networks for Semantic Segmentation, CVPR 2015  
[[Xu et al. 15]](http://proceedings.mlr.press/v37/xuc15.pdf) Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, ICML 2015  
[[Held et al. 16]](http://davheld.github.io/GOTURN/GOTURN.pdf) Learning to Track at 100 FPS with Deep Regression Networks, ECCV 2016  
***
[[Lin et al. 17]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) Focal Loss for Dense Object Detection, ICCV 2017  
[[Chen et al. 18]](https://arxiv.org/pdf/1802.02611.pdf) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, ECCV 2018  
[[Girdhar et al. 18]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Girdhar_Detect-and-Track_Efficient_Pose_CVPR_2018_paper.pdf) Detect-and-Track: Efficient Pose Estimation in Videos, CVPR 2018
-->  
