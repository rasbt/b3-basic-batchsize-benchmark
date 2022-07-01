# B3 -- Basic Batchsize Benchmark



A quick benchmark with different batch sizes that was prompted by the discussion [here](https://twitter.com/rasbt/status/1542882893181108227?s=20&t=96dUITuyaNJUfw1TWxDLng), which was in turn prompted by the [Do Batch Sizes Actually Need to be Powers of 2?](https://wandb.ai/datenzauberai/Batch-Size-Testing/reports/Do-Batch-Sizes-Actually-Need-to-be-Powers-of-2---VmlldzoyMDkwNDQx) article.



Right now, this benchmark is a [MobileNetV3 (large)](https://arxiv.org/abs/1905.02244). You can run it as follows:



**Step 1: Initial Setup**

```bash
conda create -n benchmark python=3.8
conda activate benchmark
pip install -r requirements.txt
```



**Step 2: Running the Training Script**


```python
python main.py --num_epochs 10 --batch_size 127 --mixed_precision true
```



### Additional Resources

- [Ross Wightman mentioning](https://twitter.com/wightmanr/status/1542917523556904960?s=20&t=96dUITuyaNJUfw1TWxDLng) that it might matter more for TPUs
- [Nvidia's Deep Learning Performance Documentation on matrix multiplication](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) explaining the theoretical rational behind choosing batch sizes as multiples of 8 for tensor cores



### Results




| batch size | train time | inf. time | combined  | epochs | GPU  | mixed prec. |
| ---------- | ---------- | --------- | --------- | ------ | ---- | ----------- |
| 100        | 10.50 min  | 0.15 min  | 10.78 min | 10     | V100 | Yes         |
| 127        | 9.80 min   | 0.15 min  | 10.08 min | 10     | V100 | Yes         |
| 128        | 9.78 min   | 0.10 min  | 10.07 min | 10     | V100 | Yes         |
| 129        | 9.92 min   | 0.15 min  | 10.20 min | 10     | V100 | Yes         |
| 156        | 9.38 min   | 0.16 min  | 9.67 min  | 10     | V100 | Yes         |




Note that this is all from one run each. To get more reliable stats, it might be worthwhile repeating the runs many times and reporting the average + SD. However, even from the numbers above, it is probably apparant that there is only a small but barely noticeable difference between 127, 128, and 129.

**Or in other words, do you have a batch size of 128 that you would like to run but it doesn't fit into memory? It's probably okay to train that model with a batch size of 120 and 100 before scaling it down to 64 😊.**