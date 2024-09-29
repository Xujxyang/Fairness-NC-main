# Fairness_NC

## Getting Started
Each folder contains modifications to the original model, incorporating the NC3 regularization mentioned in our paper. However, this repository does not include the datasets required for each project, so you will need to download them manually from the original repository.

For [Mabel](https://github.com/princeton-nlp/MABEL), they processed their training data from SNLI and MNLI, you need to downloaded and stored under the `training` directory as `entailment_data.csv`. 

For [Adept](https://github.com/EmpathYang/ADEPT), you can download News-Commentary v15 [here](https://data.statmt.org/news-commentary/v15/documents.tgz), and then, you need to follow the original project's instructions to process the data for debiased training. 

For [Ase](https://github.com/NLPlab-skku/BERT-ASE), you need to download the WinoBias datasets available on the [corefBias](https://github.com/uclanlp/corefBias).

For [Bec](https://github.com/marionbartl/gender-bias-BERT), just get started!

After completing the above steps, you can easily follow the instructions from each project's repository to debias the respective models. That said, you will need to modify the corresponding code files in the training commands.

**Mabel:** Use our `train.py` and ensure that the updates from `trainer_nc.py` and `model_nc.py` are applied.

**Adept:** You can reproduce the original paper's results based on `debias_original.py`.

**Ase:** Use our `finetune_both_nc.py` instead of original `finetune_both.py`.

**Bec:** Use our `main_nc.py` instead of original `main.py`.

## Acknowledgement
[linguistic-collapse](https://github.com/rhubarbwu/linguistic-collapse)
