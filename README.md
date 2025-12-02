<p align="center">

<img src="assets/home.png" alt="OmniSafeBench-MM"/>

</p>

<div align="center">

# 🗡️ OmniSafeBench-MM: A Unified Benchmark and Toolbox for Multimodal Jailbreak Attack–Defense Evaluation 🛡️

</div>

## 📰 News 📰
* ***[2024/02/06] 🚀 [Initial release of OmniSafeBench-MM with 13 attack methods and 15 defense strategies]()*** 🚀




## 🗡️ What is OmniSafeBench-MM 🛡️
OmniSafeBench-MM is a unified benchmark and open-source toolbox for evaluating multimodal jailbreak attacks and defenses in Large Vision–Language Models (MLLMs). It integrates a large-scale dataset spanning 8–9 major risk domains and 50 fine-grained categories, supports three real-world prompt types (consultative, imperative, declarative), and implements 13 representative attack methods and 15 defense strategies in a modular pipeline. Beyond traditional ASR, it introduces a three-dimensional evaluation protocol measuring harmfulness, intent alignment, and response detail, enabling fine-grained safety–utility analysis. Tested across 18 open-source and closed-source MLLMs, OmniSafeBench-MM provides a comprehensive, reproducible, and extensible platform for benchmarking multimodal safety.



## 🌐 Overview 🌐
<div align="center">

<img src="assets/overview.png" alt="HarmBench Evaluation Pipeline"/>

</div>
**Overview of OmniSafeBench-MM**.
The benchmark unifies multi-modal jailbreak attack–defense evaluation, 13 attack and 15 defense methods, and a three-dimensional scoring protocol measuring harmfulness, alignment, and detail.

## 🗡️ Integrated Attack Methods
| Name | Title                                                                                                                       | Paper | Code | Classification |
|------|-----------------------------------------------------------------------------------------------------------------------------|-------|------|----------------|
| FigStep / FigStep-Pro | FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts                                           | [link](https://arxiv.org/abs/2311.05608) | [link](https://github.com/ThuCCSLab/FigStep) | Black-box — Structured visual carriers |
| QR-Attack (MM-SafetyBench) | MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models                                       | [link](https://arxiv.org/abs/2311.17600) | [link](https://github.com/isXinLiu/MM-SafetyBench) | Black-box — Structured visual carriers |
| MML | Jailbreak Large Vision-Language Models Through Multi-Modal Linkage                                                          | [link](https://aclanthology.org/2025.acl-long.74/) | [link](https://github.com/wangyu-ovo/MML) | Black-box — Structured visual carriers |
| CS-DJ | Distraction is All You Need for Multimodal Large Language Model Jailbreaking                                                | [link](https://arxiv.org/abs/2502.10794) | [link](https://github.com/TeamPigeonLab/CS-DJ) | Black-box — OOD (attention / distribution manipulation) |
| SI-Attack | Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency                                                     | [link](https://arxiv.org/abs/2501.04931) | [link](https://github.com/zhaoshiji123/SI-Attack) | Black-box — OOD (shuffle / attention inconsistency) |
| JOOD | Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy                                   | [link](https://arxiv.org/abs/2503.20823) | [link](https://github.com/naver-ai/JOOD) | Black-box — OOD (OOD strategy) |
| HIMRD | Heuristic-Induced Multimodal Risk Distribution (HIMRD) Jailbreak Attack                                                     | [link](https://arxiv.org/abs/2412.05934) | [link](https://github.com/MaTengSYSU/HIMRD-jailbreak) | Black-box — OOD / risk distribution |
| HADES | Images are Achilles’ Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models | [link](https://arxiv.org/abs/2403.09792) | [link](https://github.com/AoiDragon/HADES) | Black-box — Structured visual carriers |
| BAP | Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt (BAP)                                                      | [link](https://arxiv.org/abs/2406.04031) | [link](https://github.com/NY1024/BAP-Jailbreak-Vision-Language-Models-via-Bi-Modal-Adversarial-Prompt) | White-box — Cross-modal |
| visual_adv | Visual Adversarial Examples Jailbreak Aligned Large Language Models                                                         | [link](https://ojs.aaai.org/index.php/AAAI/article/view/30150) | [link](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models) | White-box — Single-modal |
| VisCRA | VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models                                   | [link](https://arxiv.org/abs/2505.19684) | [link](https://github.com/DyMessi/VisCRA) | Black-box — Structured visual carriers |
| UMK | White-box Multimodal Jailbreaks Against Large Vision-Language Models (Universal Master Key, UMK)                            | [link](https://arxiv.org/abs/2405.17894) | [link](https://github.com/roywang021/UMK) | White-box — Cross-modal |
| PBI-Attack | Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization (PBI-Attack)                          | [link](https://aclanthology.org/2025.emnlp-main.32.pdf) | [link](https://github.com/Rosy0912/PBI-Attack) | Black-box — Query optimization & transfer |
| ImgJP / DeltaJP | Jailbreaking Attack against Multimodal Large Language Model (imgJP / deltaJP)                                               | [link](https://arxiv.org/abs/2402.02309) | [link](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model) | White-box — Single-modal |
| JPS | JPS: Jailbreak Multimodal Large Language Models with Collaborative Visual Perturbation and Textual Steering                 | [link](https://arxiv.org/abs/2508.05087) | [link](https://github.com/thu-coai/JPS) | White-box — Cross-modal |


## 🛡️Integrated Defense Methods
| Name | Title                                                                                                              | Venue | Paper | Code | Classification |
|---|--------------------------------------------------------------------------------------------------------------------|---|---|---|---|
| JailGuard | JailGuard: A Universal Detection Framework for Prompt-based Attacks on LLM Systems                                 | arXiv | [link](https://arxiv.org/abs/2312.10766) | [link](https://github.com/shiningrain/JailGuard) | Off-Model — Input pre-processing |
| MLLM-Protector | MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance                                                 | EMNLP 2024 | [link](https://arxiv.org/abs/2401.02906) | [link](https://github.com/pipilurj/MLLM-protector) | Off-Model — Output post-processing |
| ECSO | Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation                                | ECCV 2024 | [link](https://arxiv.org/abs/2403.09572) | [link](https://gyhdog99.github.io/projects/ecso/) | Off-Model — Input pre-processing |
| ShieldLM | ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable Safety Detectors                                | EMNLP 2024 | [link](https://arxiv.org/abs/2402.16444) | [link](https://github.com/thu-coai/ShieldLM) | Off-Model — Output post-processing |
| AdaShield | AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting | ECCV 2024 | [link](https://arxiv.org/abs/2403.09513) | [link](https://github.com/rain305f/AdaShield) | Off-Model — Input pre-processing |
| Uniguard | UNIGUARD: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models            | ECCV 2024 | [link](https://arxiv.org/abs/2411.01703) | [link](https://anonymous.4open.science/r/UniGuard/README.md) | Off-Model — Input pre-processing |
| DPS | Defending LVLMs Against Vision Attacks Through Partial-Perception Supervision                                      | ICML 2025 | [link](https://arxiv.org/abs/2412.12722) | [link](https://github.com/tools-only/DPS) | Off-Model — Input pre-processing |
| CIDER | Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models                    | EMNLP 2024 | [link](https://arxiv.org/abs/2407.21659) | [link](https://github.com/PandragonXIII/CIDER) | Off-Model — Input pre-processing |
| GuardReasoner-VL | GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning                                                       | ICML 2025 | [link](https://arxiv.org/abs/2505.11049) | [link](https://github.com/yueliu1999/GuardReasoner-VL) | Off-Model — Input pre-processing |
| Llama-Guard-4 | Llama Guard 4                                                                                                      | Model Card | [link](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) | [link](https://huggingface.co/meta-llama/Llama-Guard-4-12B) | Off-Model — Input pre-processing |
| QGuard | QGuard: Question-based Zero-shot Guard for Multi-modal LLM Safety                                                  | arXiv | [link](https://arxiv.org/abs/2506.12299) | [link](https://github.com/taegyeong-lee/QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety) | Off-Model — Input pre-processing |
| LlavaGuard | LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models                                | ICML 2025 | [link](https://arxiv.org/abs/2406.05113) | [link](https://github.com/ml-research/LlavaGuard) | Off-Model — Input pre-processing |
| Llama-Guard-3 | Llama Guard 3                                                                                                      | Model Card | [link](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/) | [model link](https://huggingface.co/meta-llama/Llama-Guard-3-8B) | Off-Model — Output post-processing |
| HiddenDetect | HiddenDetect: Detecting Jailbreak Attacks against Multimodal Large Language Models via Monitoring Hidden States    | ACL 2025 | [link](https://arxiv.org/abs/2502.14744) | [link](https://github.com/leigest519/HiddenDetect) | On-Model — Inference process intervention |
| CoCA | CoCA: Regaining Safety-awareness of Multimodal Large Language Models with Constitutional Calibration               | COLM 2024 | [link](https://arxiv.org/abs/2409.11365) | Null | On-Model — Inference process intervention |
| VLGuard | Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models                                | ICML 2024 | [link](https://arxiv.org/abs/2402.02207) | [link](https://github.com/ys-zong/VLGuard) | On-Model — Intrinsic model alignment |
> More methods are coming soon!!
