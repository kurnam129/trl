---
base_model:
- trl-internal-testing/tiny-random-LlamaForCausalLM
library_name: transformers
tags:
- mergekit
- merge

---
# merged

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the [linear](https://arxiv.org/abs/2203.05482) merge method.

### Models Merged

The following models were included in the merge:
* [trl-internal-testing/tiny-random-LlamaForCausalLM](https://huggingface.co/trl-internal-testing/tiny-random-LlamaForCausalLM)
* trained_dir/checkpoint-3

### Configuration

The following YAML configuration was used to produce this model:

```yaml
dtype: float16
merge_method: linear
slices:
- sources:
  - layer_range: [0, 2]
    model: trained_dir/checkpoint-3
    parameters:
      weight: 0.5
  - layer_range: [0, 2]
    model: trl-internal-testing/tiny-random-LlamaForCausalLM
    parameters:
      weight: 0.5
```
