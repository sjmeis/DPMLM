# DP-MLM
This is the code and package repository for the ACL Findings paper: *DP-MLM: Differentially Private Text Rewriting Using Masked Language Models*

## Setup
### Installation
You can install the package directly using:

```bash
pip install dpmlm
```

Optionally, you can install from source. In this repository, you will find a `requirements.txt` file, which contains all necessary Python dependencies.

Otherwise, there are one other include file for replication of the paper, which is easily importable and reusable:
- `LLMDP.py`: implementations of both `DP-Paraphrase` and `DP-Prompt`. Note that for `DP-Prompt`, you will need to download the corresponding LMs, i.e., from Hugging Face.

### Resource Bootstrapping
Before running the mechanism, you need to download the necessary NLTK libraries:

```python
from DPMLM.utils import setup_resources

setup_resources()
```

## Usage of DP-MLM
The core logic resides in the DPMLM class. You can now initialize it with custom calibration bounds to ensure the DP privatization is tuned to your specific model (and bounding strategy).

```python
from DPMLM import DPMLM
from DPMLM.utils import calculate_logit_bounds

# 1. (Optional) Calibrate bounds for your specific model (e.g., RoBERTa)
bounds = calculate_logit_bounds("FacebookAI/roberta-base")

# 2. Instantiate the mechanism
M = DPMLM(MODEL="FacebookAI/roberta-base", calibration=bounds, bound_strategy=None)

# 3. Rewrite text
private_text = M.dpmlm_rewrite("Hello world, this is a private text.", epsilon=25)
```

If you want to set a bounding strategy for the clip bounds (beyond simple min/max selection), you can do so by passing a lambda function:

```python
# strategy as used in the paper
strategy = lamba mean, std, low, high: (mean, mean + 4*std)
M = DPMLM(MODEL="FacebookAI/roberta-base", calibration=bounds, bound_strategy=strategy)
```

### DP-MLM Batched Mode
For longer documents, the batched mode provides significant performance increases by parallelizing masked token predictions on the GPU.

To use batching, simply run:

```python
M.dpmlm_rewrite_batch("Large document text...", epsilon=25, batch_size=16)
```

Depending on your setup, you may need to tweak the `batch_size` parameter for the most optimal performance gains.

### Input Document Length
As of the newest 2025 release, `DP-MLM` no longer has the shortcoming of the 512 token context window (256 with concatentation), which was due to the limitations of MLM context windows.

Now, `DP-MLM` operates with a *sliding window*, where the maximum context is given, centered around the target word to be privatized. Thus, `DP-MLM` now works on arbitrarily long documents!

## Usage of other evaluated models
`M = LLMDP.DPPrompt()`

`M.privatize("hello world", epsilon=100)`

## Important notes
In order to use `LLMDP.DPParaphrase`, you must download the fine-tuned model directory.
This can be found at the following link: [Model](https://drive.google.com/drive/folders/1w_6MHQEw9LGkOHx_K1tc6t9djzrprITp?usp=sharing)

## Citation
Please consider citing the original work that introduced `DP-MLM`. Thank you!

```
@inproceedings{meisenbacher-etal-2024-dp,
    title = "{DP}-{MLM}: Differentially Private Text Rewriting Using Masked Language Models",
    author = "Meisenbacher, Stephen  and
      Chevli, Maulik  and
      Vladika, Juraj  and
      Matthes, Florian",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.554/",
    doi = "10.18653/v1/2024.findings-acl.554",
    pages = "9314--9328"
}
```
