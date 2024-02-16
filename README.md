# DPMLM
## Usage of DP-MLM
`M = DPMLM.DPMLM()`
`M.dpmlm_rewrite("hello world", epsilon=100)`

## Usage of other evaluated models
`M = LLMDP.DPPrompt()`
`M.privatize("hello world", epsilon=100)`

In order to use `LLMDP.DPParaphrase`, you must download the fine-tuned model directory.
This can be found at the following link: [Model](https://drive.google.com/drive/folders/1w_6MHQEw9LGkOHx_K1tc6t9djzrprITp?usp=sharing)