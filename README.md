# B3 -- Basic Batchsize Benchmark



A quick benchmark with different batch sizes that was prompted by the discussion [here](https://twitter.com/rasbt/status/1542882893181108227?s=20&t=96dUITuyaNJUfw1TWxDLng), which was in turn prompted by the [Do Batch Sizes Actually Need to be Powers of 2?](https://wandb.ai/datenzauberai/Batch-Size-Testing/reports/Do-Batch-Sizes-Actually-Need-to-be-Powers-of-2---VmlldzoyMDkwNDQx) article.



Right now, this benchmark is a [MobileNetV3 (large)](https://arxiv.org/abs/1905.02244). You can run it as follows:



**Step 1: Initial Setup**

```bash
git clone https://github.com/rasbt/b3-basic-batchsize-benchmark.git
cd b3-basic-batchsize-benchmark
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




| batch size | train time | inf. time  | epochs | GPU  | mixed prec. |
| ---------- | ---------- | --------- | ------ | ---- | ----------- |
| 100        | 10.50 min  | 0.15 min  | 10     | V100 | Yes         |
| 127        | 9.80 min   | 0.15 min  | 10     | V100 | Yes         |
| 128        | 9.78 min   | 0.15 min  | 10     | V100 | Yes         |
| 129        | 9.92 min   | 0.15 min  | 10     | V100 | Yes         |
| 156        | 9.38 min   | 0.16 min  | 10     | V100 | Yes         |
|            |            |           |        |      |             |
| 511        | 8.74 min   | 0.17 min  | 10     | V100 | Yes         |
| 512        | 8.71 min   | 0.17 min  | 10     | V100 | Yes         |
| 513        | 8.72 min   | 0.17 min  | 10     | V100 | Yes         |


Below, I trained the same neural network using 4 V100 GPUs with the distributed data parallel strategy:

```bash
python main.py --num_epochs 10 --batch_size 255 --mixed_precision true --num_workers 4 --strategy ddp
```

| batch size | train time | epochs | GPU    | mixed prec. |
| ---------- | ---------- | ------ | ------ | ----------- |
| 255        |  2.95 min  |  10    | 4xV100 | Yes         |
| 256        |  2.87 min  |  10    | 4xV100 | Yes         |
| 257        |  2.86 min  |  10    | 4xV100 | Yes         |

Note that I removed the inference time (here: evaluation on the test set) from this table, because in practice, you would still use a single V100 for inference purposes. 




Note that this is all from one run each. To get more reliable stats, repeating the runs many times and reporting the average + SD might be worthwhile. However, even from the numbers above, it is probably apparent that there is only a small but barely noticeable difference between 127, 128, and 129.



**Or in other words, do you have a batch size of 128 that you would like to run, but it doesn't fit into memory? It's probably okay to train that model with a batch size of 120 and 100 before scaling it down to 64** 😊.

