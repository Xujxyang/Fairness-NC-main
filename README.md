# Fairness_NC

## Getting Started
Each folder contains modifications to the original model, incorporating the NC3 regularization mentioned in our paper. However, this repository does not include the datasets required for each project, so you will need to download them manually from the original repository.

For [mabel](https://github.com/princeton-nlp/MABEL), they processed their training data from SNLI and MNLI, you need to downloaded and stored under the `training` directory as `entailment_data.csv`. 

For [adept](https://github.com/EmpathYang/ADEPT), you can download News-Commentary v15 [here](https://data.statmt.org/news-commentary/v15/documents.tgz), and then, you need to follow the original project's instructions to process the data for training. 

For [bert-ase](https://github.com/NLPlab-skku/BERT-ASE), you need to download the WinoBias datasets available on the [corefBias](https://github.com/uclanlp/corefBias).

For [bec](https://github.com/marionbartl/gender-bias-BERT), just get started!

After completing the above steps, you can easily follow the instructions from each project's repository to train the respective models.

## Acknowledgement
[linguistic-collapse](https://github.com/rhubarbwu/linguistic-collapse)
