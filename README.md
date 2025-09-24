# [NeurIPS 2025] FlashMoE: Fast Distributed MoE in a Single Kernel

This is the repository for the paper: </br>

> [FlashMoE: Fast Distributed MoE in a Single Kernel](https://arxiv.org/abs/2506.04667).
>
> Osayamen Jonathan Aimuyo, Byungsoo Oh, Rachee Singh


Check out our [website](https://flash-moe.github.io)!


## Abstract
The computational sparsity of Mixture-of-Experts (MoE) models enables sub-linear growth in compute cost as model size increases, thus offering a scalable path to training massive neural networks. However, existing implementations suffer from <em>low GPU utilization</em>, <em>significant latency overhead</em>, and a fundamental <em>inability to leverage task locality</em>, primarily due to CPU-managed scheduling, host-initiated communication, and frequent kernel launches. To overcome these limitations, we develop FlashMoE, a fully GPU-resident MoE operator that fuses expert computation and inter-GPU communication into a <em>single persistent GPU kernel</em>. FlashMoE enables fine-grained pipelining of dispatch, compute, and combine phases, eliminating launch overheads and reducing idle gaps. Unlike existing work, FlashMoE obviates bulk-synchronous collectives for one-sided, device-initiated, inter-GPU (R)DMA transfers, thus unlocking <em>payload efficiency</em>, where we eliminate bloated or redundant network payloads in sparsely activated layers. When evaluated on a single 8-H100 GPU node with MoE models having up to 128 experts and 16K token sequences, FlashMoE achieves up to <strong>9</strong>× higher GPU utilization, <strong>6</strong>× lower latency, <strong>5.7</strong>× higher throughput, and <strong>4</strong>× better overlap efficiency compared to state-of-the-art baselines, despite using FP32 while baselines use FP16. FlashMoE demonstrates that principled GPU kernel-hardware co-design is key to unlocking the performance ceiling of large-scale distributed ML workloads.

## Citation

```
@article{aimuyo2025flashmoe,
  title={FlashMoE: Fast Distributed MoE in a Single Kernel},
  author={Aimuyo, Osayamen Jonathan and Oh, Byungsoo and Singh, Rachee},
  journal={Advances in Neural Information Processing Systems},
  year={2025},
}
```
