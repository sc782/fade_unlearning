# FADE: Functional Alignment for Distributional Equivalence

FADE is a metric for evaluating machine unlearning. It compares an unlearned model u
with a retain-only oracle r -- a model trained the same way but without the forget
data -- and measures how far apart their output distributions are on the forget
inputs, as a symmetric KL divergence:

```
FADE = KL(r || u) + KL(u || r)
```

Each direction is estimated on outputs sampled from the model it is taken under:
KL(r || u) on r's outputs, KL(u || r) on u's. FADE is 0 only when the two models'
output distributions on the forget inputs coincide. Independently trained oracles are
not identical either, so FADE between two retain-only oracles (different seeds) gives
the floor a well-unlearned model can be expected to reach.

How the per-sample KL is computed depends on the model family:

| | [`fade-tofu/`](fade-tofu/) | [`fade-unlearncanvas/`](fade-unlearncanvas/) |
|---|---|---|
| models | Llama-3.1-8B (language model) | Stable Diffusion v1.5 (text-to-image diffusion) |
| benchmark | [TOFU](https://locuslab.github.io/tofu/): forget fictitious authors | [UnlearnCanvas](https://github.com/OPTML-Group/UnlearnCanvas): forget an artistic style |
| samples | answers to the forget questions | images of the forget style |
| per-sample KL | exact next-token KLs summed along the answer (Rao-Blackwellized estimator), in nats per answer | upper bound: closed-form KLs between the two models' guided sampling steps, summed along the sampling chain, in nats per latent dimension |
| unlearning methods | GA, GD, NPO, DPO (refusal fine-tuning), IHL | CA, EraseDiff, ESD, SalUn, Scissorhands |

Each subdirectory is a self-contained project with its own environment, scripts,
published checkpoints and README.

## License

Code in this repository is MIT licensed -- see [LICENSE](LICENSE). The published
checkpoints are fine-tunes of third-party models and carry those models' terms
regardless of the code license: Llama 3.1 Community License for `fade-tofu`, and
Stable Diffusion's CreativeML OpenRAIL-M for `fade-unlearncanvas` (see
`fade-unlearncanvas/LICENSE`). Vendored third-party code keeps its own license notices.

## Citation

```
@article{cho2025referencespecific,
  title={Reference-Specific Unlearning Metrics Can Hide the Truth: A Reality Check},
  author={Cho, Sungjun and Hwang, Dasol and Sala, Frederic and Hwang, Sangheum and Cho, Kyunghyun and Cha, Sungmin},
  journal={arXiv preprint arXiv:2510.12981},
  year={2025}
}
```
