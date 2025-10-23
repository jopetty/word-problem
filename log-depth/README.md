# Transformer depth and state tracking

[*A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers*](https://arxiv.org/abs/2503.03961)<br/>
William Merrill, Ashish Sabharwal

To appear at NeurIPS 2025.

First use log-depth/generate-data.sh to generate depth and width data to train on.
Then use log-depth/train-by-depth.sh and log-depth/train-by-width.sh to replicate the depth and width experiments in Figure 2.
Then you can download the results from W&B and adapt src/plot.py to plot depth and width as a function of *n*.
