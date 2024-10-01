# Fairness_NC

## Collapsed Language Models Promote Fairness
This repository contains the code for our paper, ["Collapsed Language Models Promote Fairness"](). 

To mitigate societal biases implicitly encoded in recent successful pretrained language models, a diverse array of approaches have been proposed to encourage model fairness, focusing on prompting, data augmentation, regularized fine-tuning, and more.
Despite the development, it is nontrivial to reach a principled understanding of fairness and an effective algorithm that can consistently debias language models.
In this work, by rigorous evaluations of Neural Collapse -- a learning phenomenon happen in last-layer representations and classifiers in deep networks -- on fairness-related words, we find that debiased language models exhibit collapsed alignment between token representations and word embeddings.
More importantly, this observation inspires us to design a principled fine-tuning method that can effectively improve fairness in a wide range of debiasing methods, while still preserving the performance of language models on standard natural language understanding tasks.


## Getting Started
Each folder contains modifications to the original model, incorporating the NC3 regularization mentioned in our paper. However, this repository does not include the datasets required for each project, so you will need to download them manually from the original repository.

For [Mabel](https://github.com/princeton-nlp/MABEL), they processed their training data from SNLI and MNLI, you need to downloaded and stored under the `training` directory as `entailment_data.csv`. 

For [Adept](https://github.com/EmpathYang/ADEPT), you can download News-Commentary v15 [here](https://data.statmt.org/news-commentary/v15/documents.tgz), and then, you need to follow the original project's instructions to process the data for debiased training. 

For [ASE](https://github.com/NLPlab-skku/BERT-ASE), you need to download the WinoBias datasets available on the [corefBias](https://github.com/uclanlp/corefBias).

For [BEC](https://github.com/marionbartl/gender-bias-BERT), just get started!

After completing the above steps, you can easily follow the instructions from each project's repository to debias the respective models. That said, you will need to modify the corresponding code files in the training commands.

**Mabel:** Use our `train.py` and ensure that the updates from `trainer_nc.py` and `model_nc.py` are applied.

**Adept:** You can reproduce the original paper's results based on `debias_original.py`.

**ASE:** Use our `finetune_both_nc.py` instead of original `finetune_both.py`.

**BEC:** Use our `main_nc.py` instead of original `main.py`.


## Model list
**Bert-base-uncased:**
|        Model types       | ICAT ⬆️ |
|:-------------------------------|:------|
| [Mabel + (U)NC_3](https://drive.google.com/drive/folders/1XVFYzuMzzCTVZodkQMfIiikD_-eCfrvY?usp=sharing) | 74.55 | 
| [ASE + (U)NC_3](https://drive.google.com/drive/folders/1ml0hZekb1q2ZJiTAScJfzkS0-G57AuhZ?usp=sharing) |  73.37 |
| [BEC + (U)NC_3](https://drive.google.com/drive/folders/1XLZwizJrjusK8igyJNdHDW4QqBd4196r?usp=sharing) |  72.38 |

**WinoBias:**
For convenience, we directly provide the checkpoint for the WinoBias downstream task (using the BERT-base-cased model for this task).
|        Model types       | TPR-1 ⬇️ | TPR-2 ⬇️ |
|:-------------------------------|:------|:------|
| [Mabel + (U)NC_3](https://drive.google.com/drive/folders/1zKvoZQo_UXVK7bAzs0YDGoTnbWlEg3M5?usp=sharing) | 16.93 | 1.97 | 
| [ASE + (U)NC_3](https://drive.google.com/drive/folders/1OTRrGoZ_fdnkSPrdRkGF2SoUTbdmB9Lp?usp=sharing) |  27.14 | 8.16 | 
| [BEC + (U)NC_3](https://drive.google.com/drive/folders/1c4lU5vVMC2G7pgY5-I6LVQQf7nDVHdvG?usp=sharing) |  21.91 | 6.90 | 



## Acknowledgement
[linguistic-collapse](https://github.com/rhubarbwu/linguistic-collapse)
