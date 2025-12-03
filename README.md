
<div align="center">

# 🗡️ OmniSafeBench-MM: A Unified Benchmark and Toolbox for Multimodal Jailbreak Attack–Defense Evaluation 🛡️

</div>

<p align="center">

<img src="assets/home.png" alt="OmniSafeBench-MM"/>

</p>


## 📰 News 📰
* ***[2025/12/06] 🚀 [Initial release of OmniSafeBench-MM with 13 attack methods and 15 defense strategies]()*** 🚀




## 🗡️ What is OmniSafeBench-MM 🛡️
OmniSafeBench-MM is a unified benchmark and open-source toolbox for evaluating multimodal jailbreak attacks and defenses in Large Vision–Language Models (MLLMs). It integrates a large-scale dataset spanning 8–9 major risk domains and 50 fine-grained categories, supports three real-world prompt types (consultative, imperative, declarative), and implements 13 representative attack methods and 15 defense strategies in a modular pipeline. Beyond traditional ASR, it introduces a three-dimensional evaluation protocol measuring harmfulness, intent alignment, and response detail, enabling fine-grained safety–utility analysis. Tested across 18 open-source and closed-source MLLMs, OmniSafeBench-MM provides a comprehensive, reproducible, and extensible platform for benchmarking multimodal safety.



## 🌐 Overview 🌐
<div align="center">

<img src="assets/overview.png" alt="HarmBench Evaluation Pipeline"/>

</div>

**Overview of OmniSafeBench-MM**.
The benchmark unifies multi-modal jailbreak attack–defense evaluation, 13 attack and 15 defense methods, and a three-dimensional scoring protocol measuring harmfulness, alignment, and detail.

## 🗡️ Integrated Attack Methods
|            Name            |                            Title                             |   Venue    |                         Paper                          |                             Code                             |
| :------------------------: | :----------------------------------------------------------: | :--------: | :----------------------------------------------------: | :----------------------------------------------------------: |
|   FigStep / FigStep-Pro    | FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts | AAAI 2025  |            https://arxiv.org/abs/2311.05608            |             https://github.com/ThuCCSLab/FigStep             |
| QR-Attack (MM-SafetyBench) | MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models | ECCV 2024  |            https://arxiv.org/abs/2311.17600            |          https://github.com/isXinLiu/MM-SafetyBench          |
|            MML             | Jailbreak Large Vision-Language Models Through Multi-Modal Linkage |  ACL 2025  |       https://aclanthology.org/2025.acl-long.74/       |              https://github.com/wangyu-ovo/MML               |
|           CS-DJ            | Distraction is All You Need for Multimodal Large Language Model Jailbreaking | CVPR 2025  |            https://arxiv.org/abs/2502.10794            |            https://github.com/TeamPigeonLab/CS-DJ            |
|         SI-Attack          | Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency | ICCV 2025  |            https://arxiv.org/abs/2501.04931            |          https://github.com/zhaoshiji123/SI-Attack           |
|            JOOD            | Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy | CVPR 2025  |            https://arxiv.org/abs/2503.20823            |               https://github.com/naver-ai/JOOD               |
|           HIMRD            | Heuristic-Induced Multimodal Risk Distribution (HIMRD) Jailbreak Attack |  ICCV 2025 |            https://arxiv.org/abs/2412.05934            |        https://github.com/MaTengSYSU/HIMRD-jailbreak         |
|           HADES            | Images are Achilles’ Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking MLLMs | ECCV 2024 |            https://arxiv.org/abs/2403.09792            |              https://github.com/AoiDragon/HADES              |
|            BAP             | Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt (BAP) | TIFS 2025 |            https://arxiv.org/abs/2406.04031            | https://github.com/NY1024/BAP-Jailbreak-Vision-Language-Models-via-Bi-Modal-Adversarial-Prompt |
|         visual_adv         | Visual Adversarial Examples Jailbreak Aligned Large Language Models | AAAI 2024  | https://ojs.aaai.org/index.php/AAAI/article/view/30150 | https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models |
|           VisCRA           | VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models | EMNLP 2025 |            https://arxiv.org/abs/2505.19684            |              https://github.com/DyMessi/VisCRA               |
|            UMK             | White-box Multimodal Jailbreaks Against Large Vision-Language Models (Universal Master Key) | ACMMM 2024 |            https://arxiv.org/abs/2405.17894            |              https://github.com/roywang021/UMK               |
|         PBI-Attack         | Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization | EMNLP 2025 |    https://aclanthology.org/2025.emnlp-main.32.pdf     |            https://github.com/Rosy0912/PBI-Attack            |
|      ImgJP / DeltaJP       | Jailbreaking Attack against Multimodal Large Language Models | arXiv 2024 |            https://arxiv.org/abs/2402.02309            | https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model |
|            JPS             | JPS: Jailbreak Multimodal Large Language Models with Collaborative Visual Perturbation and Textual Steering | ACMMM 2025 |            https://arxiv.org/abs/2508.05087            |               https://github.com/thu-coai/JPS                |


## 🛡️Integrated Defense Methods
|       Name        |                                                        Title                                                         |   Venue    |                                       Paper                                       |                                                   Code                                                    | 
|:-----------------:|:--------------------------------------------------------------------------------------------------------------------:|:----------:|:---------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
|     JailGuard     |                  JailGuard: A Universal Detection Framework for Prompt-based Attacks on LLM Systems                  | TOSEM2025  |                     [link](https://arxiv.org/abs/2312.10766)                      |                             [link](https://github.com/shiningrain/JailGuard)                              |
|  MLLM-Protector   |                          MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance                          | EMNLP2024  |                     [link](https://arxiv.org/abs/2401.02906)                      |                            [link](https://github.com/pipilurj/MLLM-protector)                             |
|       ECSO        |                 Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation                  |  ECCV2024  |                     [link](https://arxiv.org/abs/2403.09572)                      |                             [link](https://gyhdog99.github.io/projects/ecso/)                             | 
|     ShieldLM      |                 ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable Safety Detectors                  | EMNLP2024  |                     [link](https://arxiv.org/abs/2402.16444)                      |                               [link](https://github.com/thu-coai/ShieldLM)                                | 
|     AdaShield     |  AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting  |  ECCV2024  |                     [link](https://arxiv.org/abs/2403.09513)                      |                               [link](https://github.com/rain305f/AdaShield)                               |
|     Uniguard      |       UNIGUARD: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models        |  ECCV2024  |                     [link](https://arxiv.org/abs/2411.01703)                      |                       [link](https://anonymous.4open.science/r/UniGuard/README.md)                        | 
|        DPS        |                    Defending LVLMs Against Vision Attacks Through Partial-Perception Supervision                     |  ICML2025  |                     [link](https://arxiv.org/abs/2412.12722)                      |                                 [link](https://github.com/tools-only/DPS)                                 | 
|       CIDER       |           Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models            | EMNLP2024  |                     [link](https://arxiv.org/abs/2407.21659)                      |                              [link](https://github.com/PandragonXIII/CIDER)                               | 
| GuardReasoner-VL  |                             GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning                             |  ICML2025  |                     [link](https://arxiv.org/abs/2505.11049)                      |                          [link](https://github.com/yueliu1999/GuardReasoner-VL)                           | 
|   Llama-Guard-4   |                                                    Llama Guard 4                                                     | Model Card | [link](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/)  |                        [link](https://huggingface.co/meta-llama/Llama-Guard-4-12B)                        | 
|      QGuard       |                          QGuard: Question-based Zero-shot Guard for Multi-modal LLM Safety                           |   ArXiv    |                     [link](https://arxiv.org/abs/2506.12299)                      | [link](https://github.com/taegyeong-lee/QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety) | 
|    LlavaGuard     |                 LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models                  |  ICML2025  |                     [link](https://arxiv.org/abs/2406.05113)                      |                             [link](https://github.com/ml-research/LlavaGuard)                             |
|   Llama-Guard-3   |                                                    Llama Guard 3                                                     | Model Card | [link](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/)  |                        [link](https://huggingface.co/meta-llama/Llama-Guard-3-8B)                         | 
|   HiddenDetect    |   HiddenDetect: Detecting Jailbreak Attacks against Multimodal Large Language Models via Monitoring Hidden States    |  ACL2025   |                     [link](https://arxiv.org/abs/2502.14744)                      |                            [link](https://github.com/leigest519/HiddenDetect)                             |
|       CoCA        |         CoCA: Regaining Safety-awareness of Multimodal Large Language Models with Constitutional Calibration         |  COLM2024  |                     [link](https://arxiv.org/abs/2409.11365)                      |                                                    -                                                      | 
|      VLGuard      |                 Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models                  |  ICML2024  |                     [link](https://arxiv.org/abs/2402.02207)                      |                                [link](https://github.com/ys-zong/VLGuard)                                 | 
> More methods are coming soon!!
