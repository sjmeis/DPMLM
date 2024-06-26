# DPMLM
This is the code repository for the ACL Findings paper: *DP-MLM: Differentially Private Text Rewriting Using Masked Language Models*

## Setup
In this repository, you will find a `requirements.txt` file, which contains all necessary Python dependencies.

Otherwise, there are two main files, both of which arte easily importable and reusable:
- `DPMLM.py`: code for running the `DP-MLM` mechanism. `privatize` replaces a single token, while `dpmlm_rewrite` will rewrite an entire text.
- `LLMDP.py`: implementations of both `DP-Paraphrase` and `DP-Prompt`. Note that for `DP-Prompt`, you will need to download the corresponding LMs, i.e., from Hugging Face.

## Usage of DP-MLM
`M = DPMLM.DPMLM()`

`M.dpmlm_rewrite("hello world", epsilon=100)`

## Usage of other evaluated models
`M = LLMDP.DPPrompt()`

`M.privatize("hello world", epsilon=100)`

## Important notes
In order to use `LLMDP.DPParaphrase`, you must download the fine-tuned model directory.
This can be found at the following link: [Model](https://drive.google.com/drive/folders/1w_6MHQEw9LGkOHx_K1tc6t9djzrprITp?usp=sharing)

Also, you will need to download the wordnet 2022 corpus: `python -m wn download oewn:2022`

Finally, each code implementation sets specific clipping bounds, which was done for the purposes of comparable evaluation in the paper. These can be freely changed in the parameters, and should be experimented with for (possibly) better performance.
