---
layout: post
title: 'Paper Notebook : Efficient softmax approximation for GPUs'
date: 2019-08-28
author: yuxin.wang
color: rgb(255,210,32)
cover: 'https://i.loli.net/2019/09/04/s8RacK1hNwIOrzH.png'
tags: adaptive-softmax transformer-XL
---



# Paper Notebook : Efficient softmax approximation for GPUs

| key   | value                                                        |
| ----- | ------------------------------------------------------------ |
| paper | [Efficient softmax approximation for GPUs](papers3://publication/uuid/2DA19A0A-0FF2-4203-83AD-ECBAFD0170EA) |
| tags  | `softmax` `adaptive softmax` `transformer-XL`                |
| Code  | `torch.nn.AdaptiveLogSoftmaxWithLoss()`                      |

## Absract

> We propose an approximate strategy to efficiently train neural network based language models over very large vocabularies.

本文提出了一种在大词汇表下语言模型的一种近似训练策略，这种方法被称作 *adaptive softmax*，通过这种方法可以在基本上不降低模型性能的基础上，极大的提高训练的速度，顺带减少参数数量，使得百万词汇的语言模型成为可能。

对于不想过深了解的同学，请查看这两篇 blog，写的比下文好。

- [从字到词，大词典中文BERT模型的探索之旅 | 腾讯技术工程](https://mp.weixin.qq.com/s/OBkEsjNBJaYws8UQbZ9B0A)
-  [Adaptive Softmax | 简书](https://www.jianshu.com/p/d6d5575bccff)

## Related Work

为了更高效的 *softmax* ，前辈们已经做出了很多成果。具体见 Strategies for training large vocabulary neural language models。

## Our approach: the adaptive softmax

### Computation time model of matrix-multiplication

![](https://i.loli.net/2019/08/27/nEtjDLFGsirvHO2.png)

*softmax* 其实主要的瓶颈在矩阵的乘法上。也就是 *hidden states*  $size =  B \times d$ 以及 *word representation* $size =  d \times k$.

但是，*GPU* 的计算时间并不是完全线性的。而是像上图中描述的一样，是有一个拐点的。我们可以通过以下的算式来计算 GPU 的计算时间。其中 $c$ 和 $\lambda$ 是常数。
$$
g(k, B)=\max \left(c+\lambda k_{0} B_{0}, c+\lambda k B\right)
$$
所以，我们如果在上述乘法中的 k 太小时，实际上低效的。这提醒我们，*hierarchical softmax* 在计算的效率上将并高效，或者说没有最大化 GPU 的性能。我们因此可以选择更高效的方式来做层次化的 *softmax*, 这就是 *adaptive softmax* 。

*adaptive softmax* 其实就是将词汇表按照出现的概率将不同的词汇分到不同的簇中去，并使用 *short list* 而不是 *2-level tree* 的方式来进行预测。如果到了这里还不是很理解，你可能有以下几个问题要问。

### Q&A

#### 我们应该分成几个簇呢？

![](https://i.loli.net/2019/08/27/o9yc48tBVuGskXj.png)

我们的簇不应该太小，原因上文中已经论述过了。而最优的分簇方式可以根据上文的计算公式来确定，上图的实验也证实这样的观点。

> In practice, we thus decide to use a small number of clusters (between 2 and 5), as it usually lead to slightly better perplexity, and we empirically determine the best speed/perplexity compromise on training data.

根据论文中的实验，我们最好分成 2 到 5 簇。这样既提高了速度又使得模型性能不至于损失太多。

#### 什么是 *short list* 和 *2-level tree*?

![](https://i.loli.net/2019/08/27/gzBKdYalMrVRN4L.png)

*short list* 的概念来源于这篇 `Structured output layer neural network language model`

*2-level tree* 的概念来源于这篇 `Extensions of recurrent neural network language model`

简单来讲，*2-level tree* 是当有两个簇的时候，我先预测是 A簇还是 B簇，之后再在相应的子集中做 softmax，这是一种典型的二叉树的思想。而 *short list* 是指我先预测 $W1_{a}, W2_{a},W3_{a},W4_{a},C_b$，其中最后一个代表 B簇，而前面的都是 A簇中的词汇，这种方式并不是一个二叉树的概念，如上图所示。

#### 为什么选择 *short list* 而不是 *2-level tree*?

> Compromising between efficiency and accuracy. We observe empirically that putting all the clusters in the leaves of the tree leads to a significant drop of performance (around 5 􀀀 10% performance drop, Mikolov et al., 2011c; Zweig & Makarychev, 2013). The reason is that the probability of every word w belonging to a cluster cis multiplied by the probability of its class, i.e., it is equal to P(c j h)P(w j c; h), while attaching a frequent word directly to the root associates it directly to the probability P(w j h) making its inference sharper. For this reason, unless there is a significant difference in computation time, we favor using a short-list, over the standard 2-level hierarchical softmax.

论文中进行了较详细的论述，总结一下就是 *2-level tree* 会造成较大的性能损失，因为没有考虑词汇出现的频率分布情况。  

如果你还是不明白，可以参考下面的blog，解释的很清楚。[Adaptive Softmax | 简书](https://www.jianshu.com/p/d6d5575bccff)

## 总结

![](https://i.loli.net/2019/08/27/mfivAVKEFh8sBuD.png)

*adaptive softxmax* 可以在非常小的性能损失下，极大的提高训练的速度。具体对比如上图所示。

*adaptive softmax* 的 `pytorch` 实现可以查看 `torch.nn.AdaptiveLogSoftmaxWithLoss()`