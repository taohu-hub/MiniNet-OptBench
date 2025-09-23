
## openwebtext dataset

after running `prepare.py` (preprocess) we get:

- train.bin is ~17GB, val.bin ~8.5MB
- train has ~9B tokens (9,035,582,198)
- val has ~4M tokens (4,434,897)

this came from 8,013,769 documents in total.

references:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset


5%采样：
Total samples in OpenWebText: 8013769
Sampled 400688 documents (5.0%)
Encoding train split...
Encoding val split...
Saving train (406760492 tokens) to /DATA/disk1/qyy/optimization/nanoGPT/data/openwebtext/train_small.bin
Saving val (45234357 tokens) to /DATA/disk1/qyy/optimization/nanoGPT/data/openwebtext/val_small.bin
