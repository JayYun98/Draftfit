# DICE token ranking

Which draft tokens would receive the highest scores if the next position's sampling noise were already known?
Run the [example](../../../../examples/configs/research/dice-ranking.yaml) with
`python -m draftfit.research -c examples/configs/research/dice-ranking.yaml`.
Token 0 has the highest draft logit, but the supplied draw puts token 1 first.
To compare without noise, run the same command again with
`temperature=0 output_dir=outputs/research/dice-ranking-no-noise`.

Supply raw draft logits and one vector of standard Gumbel draws in the same full
vocabulary order. Use the same vector for prefixes predicting the same absolute
output position of a request. This CPU recipe takes the draw as data; it does
not generate or verify a sampler's random stream. Set the logit scale
`alpha` and uncertainty scale `sigma` explicitly; the example values are not
fitted parameters. The result contains probabilities in vocabulary order and
the top token IDs, with lower IDs first on exact ties.

This is an independent implementation of Eq. 3 in
[Zeng and Zanette, arXiv:2611.27891v1](https://arxiv.org/html/2611.27891v1#S3.E3).
No author code or weights were used. Scores estimate the target choice; they
are not target probabilities measured from a model. The recipe does not build
a tree, verify tokens, implement prefetching or measure serving performance.
