# Token-wise Decomposition of Autoregressive Language Model Hidden States for Analyzing Model Predictions

## Introduction
This is the code repository for the paper [Token-wise Decomposition of Autoregressive Language Model Hidden States for Analyzing Model Predictions](https://arxiv.org/pdf/2305.10614.pdf), including a modified version of the [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) repository with a version of the OPT language model (Zhang et al., 2022) that returns decomposed representations that are attributable to each input token.
The ΔLP importance measure proposed in this work is calculated using these decomposed representations.

## Setup
1) Install the following major dependencies:
- [Python](https://www.python.org) (v3.10.8 used in this work)
- [PyTorch](https://pytorch.org) (v1.13.1 used in this work)

2) Install the modified version of the Transformers repository using these commands:
```
cd huggingface
pip install -e .
```

## Calculating Decomposed Representations
The command `python main.py INPUT_FILE HUGGINGFACE_OPT_MODEL DECOMPOSITION_LENGTH` (e.g. `python main.py my_corpus.sentitems facebook/opt-125m 50`) can be used to calculate decomposed representations using the modified version of the pre-trained OPT language model.

The `INPUT_FILE` is split according to `!ARTICLE` delimiters and assigned to different batches. 

```
$ head my_corpus.sentitems
!ARTICLE
If you were to journey to the North of England, you would come to a valley that is surrounded by moors as high as mountains.
```

`DECOMPOSITION_LENGTH` sets the number of most recent input tokens for which to calculate decomposed representations.
For example, setting `DECOMPOSITION_LENGTH` to 50 will return `n_{x,L+1,i,i-49}, ..., n_{x,L+1,i,i}`.
Setting `DECOMPOSITION_LENGTH` to a sufficiently large number (e.g. 2048) will return decomposed representations for all input tokens.

Running the above command will save the following representations under the `output` directory:
- `d_{INPUT_FILE_NAME}_{OPT_MODEL}_{BATCH_NUMBER}.pt`: Decomposed representations of shape `(DECOMPOSITION_LENGTH, SEQUENCE_LENGTH, HIDDEN_STATE_SIZE)`
- `i_{INPUT_FILE_NAME}_{OPT_MODEL}_{BATCH_NUMBER}.pt`: Input tokens of shape `(1, SEQUENCE_LENGTH)`
- `o_{INPUT_FILE_NAME}_{OPT_MODEL}_{BATCH_NUMBER}.pt`: Final (undecomposed) hidden states of shape `(1, SEQUENCE_LENGTH, HIDDEN_STATE_SIZE)`
- `{OPT_MODEL}_proj_weights.pt`: Projection matrix of shape `(VOCABULARY_SIZE, HIDDEN_STATE_SIZE)`

## Calculating ΔLP
With the decomposed representations in the `output` directory, the command `python representations_to_deltalp.py {INPUT_FILE_NAME}_{OPT_MODEL}_{BATCH_NUMBER} HUGGINGFACE_OPT_MODEL > OUTPUT_FILE` (e.g. `python representations_to_deltalp.py my_corpus_opt-125m_0 facebook/opt-125m > my_corpus_opt-125m_0.deltalp`) can be used to calculate ΔLP for each decomposed input token.

```
$ head my_corpus_opt-125m_0.deltalp
my_corpus_opt-125m_0 Timestep 1/28
LP(If | </s>) = -6.854281902313232
-</s> = -inf;	Delta LP: inf
Timestep 1, argmax: -1, distance: 2

my_corpus_opt-125m_0 Timestep 2/28
LP(Ġyou | </s> If) = -1.2170559167861938
-</s> = -1.3649682998657227;	Delta LP: 0.1479123830795288
-If = -3.690932035446167;	Delta LP: 2.4738759994506836
Timestep 2, argmax: 1, distance: 1
```

## Questions
For questions or concerns, please contact Byung-Doh Oh ([oh.531@osu.edu](mailto:oh.531@osu.edu)).
