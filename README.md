<!---
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


<h2 align="center">
    <p> R-LoRA: Recurrent LoRA Adapters for Improved Performance at Low Cost </p>
</h2>

R-PEFT uses recurrent gradient-free LoRA adapters as structured noise on the weights of LoRA adapters to increase robustness and accuracy of parameter-efficient fine tuning with minimal overhead. 


## Setup

This work used a PyTorch NGC Container with the additional following packages installed:

1. fire
2. datasets
3. transformers
4. peft

These can be installed with 
```bash
pip install -r requirements.txt
```

Note: Results in the paper were obtained with PyTorch NGC Container 25.01, peft 0.17, and transformers 4.56.2. Code has been tested to work up through container 26.09, transformers 5.22, and peft 0.21 (the latest versions at the time of writing this README).

## Training (finetune.py)

This file contains some code related to prompt construction and tokenization. In this file specify model and recursion params, save locations, and different sets of data, so that different models can be trained and saved. All of our training was done on a single A100 GPU; multi-GPU training has not been tested.

Example usage for Commonsense-170k training:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model 'meta-llama/Meta-Llama-3-8b' \
    --data_path './ft-training_set/commonsense_170k.json'  \
    --output_dir './trained_models/recursive_llama3/'  \
    --batch_size 16 \
    --learning_rate 3e-4  \
    --cutoff_len 256 \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 \
    --lora_alpha 8 \
    --recursion_depth 7 \
    --use_recursive_peft

```


Example usage for MetaMath-40k training:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model 'google/gemma-7b' \
    --data_path './ft-training_set/metamath_40k.json'  \
    --output_dir './trained_models/recursive_gemma/'  \
    --batch_size 16 \
    --learning_rate 3e-4  \
    --cutoff_len 256 \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"]' \
    --lora_r 32 \
    --lora_alpha 8 \
    --recursion_depth 7 \
    --use_recursive_peft

```

## Evaluation (evaluate.py)

To evaluate the performance of the finetuned model you can use the following commands:

For commonsense reasoning tasks:
```bash
CUDA_VISIBLE_DEVICES=0 python mixed_model_commonsense_evaluate.py \
    --lora_weights './trained_models/recursive_llama3/' \
    --base_model 'meta-llama/Meta-Llama-3-8b' \
    --batch_size 32 \
    --dataset boolq # specify one of ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]

```
For math tasks:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --lora_weights './trained_models/recursive_gemma/' \
    --base_model 'google/gemma-7b' \
    --dataset 'gsm8k' # specify one of ['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP', 'MATH']

```

<!-- ## Resource Consumption

There is a table of resouce needed for different adapters, which contains Trainable Parameters, GPU RAM Usage, and Fine-tuning Time on the Arithmetic Reasoning dataset `math_10k.json`

Hyper-parameter setting: num_epochs=3, lora_r=8, lora_alpha=16, bottleneck_size=256

Models: LLaMA-13B, LLaMA-7B, BLOOM-6.7B, GPT-j-6B
Dataset: 3.2K math word problems

Hardware: 2*3090 GPUs

| Model                 | Trainable Parameters | GPU RAM Usage | Fine-tuning Time |
|-----------------------|----------------------|---------------|------------------|
| LLaMA-7B-LoRA         | 4.2M                 | 18GB          |     4h           | 
| LLaMA-7B-AdapterH     | 200M                 | 22GB          |     4h           | 
| LLaMA-7B-AdapterP     | 200M                 | 22GB          |     4h           | 
| LLaMA-7B-Parallel     | 200M                 | 22GB          |     4h           |  


## Finetune Result
There are the finetune results in different models with 4 math reasoning datasets, which contains GSM8K, AQuA, SVAMP, and MAWPS. In this table, we use the optimal configuration and placement of Prefix-Tuning, Series Adapter, LoRA, and Parallel Adapter according to the empirical study in our [paper](https://aclanthology.org/2023.emnlp-main.319/).

| Model                 | GSM8K  | AQuA   |   MAWPS  |  SVAMP | Average |
|-----------------------|--------|--------|----------|--------|---------|
| GPT-3.5               |**56.4**|**38.9**| **87.4** |**69.9**|**63.2** |
| BLOOMz-7B-Prefix	    | 13.8   |  12.5  |   47.5   |  24.1  |  24.5   |
| BLOOMz-7B-Series	    | 14.3   |  20.5  |   62.2   |  38.1  |  33.8   |
| BLOOMz-7B-Parallel	| 18.5   |  18.9  |   70.6   |  36.4  |  36.1   |
| BLOOMz-7B-LoRA	    | 17.4	 |  21.3  |   70.2   |  41.0  |  37.5   |
| GPT-j-6B-Prefix	    | 16.0	 |  14.7  |   59.2   |  31.0  |  30.2   |
| GPT-j-6B-Series	    | 19.5	 |  15.0  |   80.3   |  43.6  |  39.6   |
| GPT-j-6B-Parallel	    | 18.9	 |  17.9  |   78.2   |  41.1  |  39.0   |
| GPT-j-6B-LoRA	        | 23.0	 |  16.1  |   79.4   |  46.0  |  41.1   |
| LLaMA-7B-Prefix	    | 24.4	 |  14.2  |   63.4   |  38.1  |  35.0   |
| LLaMA-7B-Series	    | 33.3	 |  15.0  |   77.7   |  52.3  |  44.6   |
| LLaMA-7B-Parallel	    | 35.3	 |  18.1  |   82.4   |  49.6  |  46.4   |
| LLaMA-7B-LoRA	        | 37.5	 |  18.9  |   79.0   |  52.1  |  46.9   |
| LLaMA-13B-Prefix	    | 31.1	 |  15.7  |   66.8   |  41.4  |  38.8   |
| LLaMA-13B-Series	    | 44.0	 |  22.0  |   78.6   |  50.8  |  48.9   |
| LLaMA-13B-Parallel	| 43.3	 |  20.5  |   81.1   |  55.7  |  50.2   |
| LLaMA-13B-LoRA	    | 47.5	 |  18.5  |   83.6   |  54.6  |  51.1   |


There are the finetune results in different models with eight commonsense reasoning datasets.

| Model                 |  BoolQ  |  PIQA  |  SIQA  |  HellaSwag  |  WinoGrande  |  ARC-e  |  ARC-c  |  OBQA  |  Average  |
|-----------------------|---------|--------|--------|-------------|--------------|---------|---------|--------|-----------|
| ChatGPT               | **73.1**|**85.4**|  68.5  |  78.5       |  66.1        |**89.8** |**79.9** |  74.8  |  77.0     |
| BLOOMz-7B-Prefix	    |   45.6  |  53.7  |  46.3  |  26.7       |  49.5        |  52.1   |  39.7   |  44.3  |  44.7     |
| BLOOMz-7B-Series	    |   65.4  |  70.4  |  73.6  |  53.4       |  69.3        |  72.3   |  55.9   |  68.0  |  66.0     |
| BLOOMz-7B-Parallel	  |   64.1  |  71.5  |  72.1  |  52.9       |  67.0        |  70.5   |  54.7   |  69.6  |  65.3     |
| BLOOMz-7B-LoRA	      |   65.9  |  75.3  |  74.5  |  57.3       |  72.5        |  74.6   |  57.8   |  73.4  |  68.9     |
| GPT-j-6B-Prefix	      |   63.1  |  66.9  |  68.7  |  34.4       |  64.5        |  64.4   |  46.8   |  59.0  |  58.5     |
| GPT-j-6B-Series	      |   62.1  |  63.5  |  72.3  |  30.6       |  68.0        |  63.9   |  48.1   |  63.8  |  59.0     |
| GPT-j-6B-Parallel	    |   62.2  |  69.7  |  70.0  |  41.7       |  65.0        |  60.2   |  44.6   |  58.2  |  59.0     |
| GPT-j-6B-LoRA	        |   62.4  |  68.6  |  49.5  |  43.1       |  57.3        |  43.4   |  31.0   |  46.6  |  50.2     |
| LLaMA-7B-Prefix	      |   64.3  |  76.8  |  73.9  |  42.1       |  72.1        |  72.9   |  54.0   |  60.6  |  64.6     |
| LLaMA-7B-Series	      |   63.0  |  79.2  |  76.3  |  67.9       |  75.7        |  74.5   |  57.1   |  72.4  |  70.8     |
| LLaMA-7B-Parallel	    |   67.9  |  76.4  |  78.8  |  69.8       |  78.9        |  73.7   |  57.3   |  75.2  |  72.3     |
| LLaMA-7B-LoRA	        |   68.9  |  80.7  |  77.4  |  78.1       |  78.8        |  77.8   |  61.3   |  74.8  |  74.7     |
| LLaMA-13B-Prefix	    |   65.3  |  75.4  |  72.1  |  55.2       |  68.6        |  79.5   |  62.9   |  68.0  |  68.4     |
| LLaMA-13B-Series	    |   71.8  |  83.0  |  79.2  |  88.1       |  82.4        |  82.5   |  67.3   |  81.8  |  79.5     |
| LLaMA-13B-Parallel	  |   72.5  |  84.8  |  79.8  |**92.1**     |**84.7**      |  84.2   |  71.2   |**82.4**|**81.5**   |
| LLaMA-13B-LoRA	      |   72.1  |  83.5  |**80.5**|  90.5       |  83.7        |  82.8   |  68.3   |**82.4**|  80.5     |


### Adapter support matrix
This metrix shows whether different models can use LoRA,AdapterH,AdapterP,Parallel and Scaled Parallel adapters.

| Adapter      | LoRA | AdapterH | AdapterP | Parallel| Prefix Tuning	|P-Tuning|Prompt Tuning|
|--------------|-------|-------|----------|-------|-------|-------|-------|
| LLaMA        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| BLOOM        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 
| GPT-J        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| OPT          | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| GPT-2        | ✅     | 🔧Developing | 🔧Developing|🔧Developing | ✅     | ✅     | ✅     | 
| GPT-Neo      | ✅     | ✅     | ✅        | ✅    | ✅     | ✅     | ✅     | 
| GPT-NeoX-20B | ✅     | 🔧Developing | 🔧Developing|🔧Developing | ✅     | ✅     | ✅     |
| ChatGLM      | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 


### TODO List
- [x] Add AdapterH
- [x] Add AdapterP
- [x] Add Parallel Adapter
- [ ] Support More LLMs
- [ ] Support Multiple Adapter
- [ ] Support Adapter Composition
- [ ] Support Adapter Fusion


## :star: Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AGI-Edgerunners/LLM-Adapters&type=Date)](https://star-history.com/#AGI-Edgerunners/LLM-Adapters&Date)

## Citing <img src="picture.jpg" width="14px" height="14px"> LLM-Adapter

If you use <img src="picture.jpg" width="14px" height="14px"> LLM-Adapters in your publication, please cite it by using the following BibTeX entry.

```bibtex
@article{hu2023llm,
  title={LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models},
  author={Hu, Zhiqiang and Lan, Yihuai and Wang, Lei and Xu, Wanyu and Lim, Ee-Peng and Lee, Roy Ka-Wei and Bing, Lidong and Poria, Soujanya},
  journal={arXiv preprint arXiv:2304.01933},
  year={2023}
}
```
-->
## Acknowledgement

This repo was built on top of [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) and the authors would like to acknowledge their work.
