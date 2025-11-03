# OmniSafeBench-MM
A Unified Benchmark and Toolbox for Multimodal Jailbreak Attack–Defense Evaluation

## 🗡️ Integrated Attack Methods
| Name | Full paper title | Paper | Code | Classification |
|------|------------------|-------|------|----------------|
| FigStep / FigStep-Pro | FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts | [paper link](https://arxiv.org/abs/2311.05608) | [code link](https://github.com/ThuCCSLab/FigStep) | Black-box — Structured visual carriers |
| QR-Attack (MM-SafetyBench) | MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models | [paper link](https://arxiv.org/abs/2311.17600) | [code link](https://github.com/isXinLiu/MM-SafetyBench) | Black-box — Structured visual carriers |
| MML | Jailbreak Large Vision-Language Models Through Multi-Modal Linkage | [paper link](https://aclanthology.org/2025.acl-long.74/) | [code link](https://github.com/wangyu-ovo/MML) | Black-box — Structured visual carriers |
| CS-DJ | Distraction is All You Need for Multimodal Large Language Model Jailbreaking | [paper link](https://arxiv.org/abs/2502.10794) | [code link](https://github.com/TeamPigeonLab/CS-DJ) | Black-box — OOD (attention / distribution manipulation) |
| SI-Attack | Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency | [paper link](https://arxiv.org/abs/2501.04931) | [code link](https://github.com/zhaoshiji123/SI-Attack) | Black-box — OOD (shuffle / attention inconsistency) |
| JOOD | Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy | [paper link](https://arxiv.org/abs/2503.20823) | [code link](https://github.com/naver-ai/JOOD) | Black-box — OOD (OOD strategy) |
| HIMRD | Heuristic-Induced Multimodal Risk Distribution (HIMRD) Jailbreak Attack | [paper link](https://arxiv.org/abs/2412.05934) | [code link](https://github.com/MaTengSYSU/HIMRD-jailbreak) | Black-box — OOD / risk distribution |
| HADES | Images are Achilles’ Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models | [paper link](https://arxiv.org/abs/2403.09792) | [code link](https://github.com/AoiDragon/HADES) | Black-box — Structured visual carriers |
| BAP | Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt (BAP) | [paper link](https://arxiv.org/abs/2406.04031) | [code link](https://github.com/NY1024/BAP-Jailbreak-Vision-Language-Models-via-Bi-Modal-Adversarial-Prompt) | White-box — Cross-modal |
| visual_adv | Visual Adversarial Examples Jailbreak Aligned Large Language Models | [paper link](https://ojs.aaai.org/index.php/AAAI/article/view/30150) | [code link](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models) | White-box — Single-modal |
| VisCRA | VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models | [paper link](https://arxiv.org/abs/2505.19684) | [code link](https://github.com/DyMessi/VisCRA) | Black-box — Structured visual carriers |
| UMK | White-box Multimodal Jailbreaks Against Large Vision-Language Models (Universal Master Key, UMK) | [paper link](https://arxiv.org/abs/2405.17894) | [code link](https://github.com/roywang021/UMK) | White-box — Cross-modal |
| PBI-Attack | Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization (PBI-Attack) | [paper link](https://aclanthology.org/2025.emnlp-main.32.pdf) | [code link](https://github.com/Rosy0912/PBI-Attack) | Black-box — Query optimization & transfer |
| ImgJP / DeltaJP | Jailbreaking Attack against Multimodal Large Language Model (imgJP / deltaJP) | [paper link](https://arxiv.org/abs/2402.02309) | [code link](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model) | White-box — Single-modal |
| JPS | JPS: Jailbreak Multimodal Large Language Models with Collaborative Visual Perturbation and Textual Steering | [paper link](https://arxiv.org/abs/2508.05087) | [code link](https://github.com/thu-coai/JPS) | White-box — Cross-modal |


## 🛡️ Integrated Defense Methods
| Name | Full paper title | Venue | Paper | Code | Classification |
|---|---|---|---|---|---|
| JailGuard | JailGuard: A Universal Detection Framework for Prompt-based Attacks on LLM Systems | arXiv | [paper link](https://arxiv.org/abs/2312.10766) | [code link](https://github.com/shiningrain/JailGuard) | Off-Model — Input pre-processing |
| MLLM-Protector | MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance | EMNLP 2024 | [paper link](https://arxiv.org/abs/2401.02906) | [code link](https://github.com/pipilurj/MLLM-protector) | Off-Model — Output post-processing |
| ECSO | Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation | ECCV 2024 | [paper link](https://arxiv.org/abs/2403.09572) | [code link](https://gyhdog99.github.io/projects/ecso/) | Off-Model — Input pre-processing |
| ShieldLM | ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable Safety Detectors | EMNLP 2024 | [paper link](https://arxiv.org/abs/2402.16444) | [code link](https://github.com/thu-coai/ShieldLM) | Off-Model — Output post-processing |
| AdaShield | AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting | ECCV 2024 | [paper link](https://arxiv.org/abs/2403.09513) | [code link](https://github.com/rain305f/AdaShield) | Off-Model — Input pre-processing |
| Uniguard | UNIGUARD: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models | ECCV 2024 | [paper link](https://arxiv.org/abs/2411.01703) | [code link](https://anonymous.4open.science/r/UniGuard/README.md) | Off-Model — Input pre-processing |
| DPS | Defending LVLMs Against Vision Attacks Through Partial-Perception Supervision | ICML 2025 | [paper link](https://arxiv.org/abs/2412.12722) | [code link](https://github.com/tools-only/DPS) | Off-Model — Input pre-processing |
| CIDER | Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models | EMNLP 2024 | [paper link](https://arxiv.org/abs/2407.21659) | [code link](https://github.com/PandragonXIII/CIDER) | Off-Model — Input pre-processing |
| GuardReasoner-VL | GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning | ICML 2025 | [paper link](https://arxiv.org/abs/2505.11049) | [code link](https://github.com/yueliu1999/GuardReasoner-VL) | Off-Model — Input pre-processing |
| Llama-Guard-4 | Llama Guard 4 | Model Card | [document link](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) | [model link](https://huggingface.co/meta-llama/Llama-Guard-4-12B) | Off-Model — Input pre-processing |
| QGuard | QGuard: Question-based Zero-shot Guard for Multi-modal LLM Safety | arXiv | [paper link](https://arxiv.org/abs/2506.12299) | [code link](https://github.com/taegyeong-lee/QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety) | Off-Model — Input pre-processing |
| LlavaGuard | LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models | ICML 2025 | [paper link](https://arxiv.org/abs/2406.05113) | [code link](https://github.com/ml-research/LlavaGuard) | Off-Model — Input pre-processing |

> More methods are coming soon!!
