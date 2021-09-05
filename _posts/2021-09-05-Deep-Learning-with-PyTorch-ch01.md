---
toc: true
layout: post
description: A synopsys of chapter 01 of book "Deep Learning with PyTorch".
categories: [markdown]
title: Chapter 01 summary
---
# 
## How deep learning has changed the way we look !
This can be easily understood by this picture

![](/images/dl1.JPG "Deep Learning 2 Scenario")

- We need a way to ingest whatever data we have at hand.
- We somehow need to define the deep learning machine.
- We must have an automated way, training, to obtain useful representations and make the machine produce desired outputs.

## Why PyTorch
- PyTorch gives us a data type, the Tensor, to hold numbers, vectors, matrices, or arrays in general. In addition, it provides functions for operating on them.
- It provides accelerated computation using graphical processing units (GPUs), often yielding speedups in the range of 50x over doing the same calculation on a CPU.
- PyTorch provides facilities that support numerical optimization on generic mathematical expressions, which deep learning uses for training.
- One of the motivations for this capability is to provide a reliable strategy for deploying models in production.
- Moving computations from the CPU to the GPU in PyTorch doesn’t require more than an additional function call or two. The second core thing that PyTorch provides is the ability of tensors to keep track of the operations performed on them and to analytically compute derivatives of an output of a computation with respect to any of its inputs. This is used for numerical optimization, and it is provided natively by tensors by virtue of dispatching through PyTorch’s autograd engine under the hood.
- The core PyTorch modules for building neural networks are located in torch.nn, which provides common neural network layers and other architectural components. Fully connected layers, convolutional layers, activation functions, and loss functions can all be found here.

  
### A camparision to TensorFlow
TensorFlow has a robust pipeline to production, an extensive industry-wide community,
and massive mindshare. PyTorch has made huge inroads with the research and
teaching communities, thanks to its ease of use, and has picked up momentum since,
as researchers and graduates train students and move to industry.

## How PyTorch supports deep learning projects.
- First we need to physically get the data, most often from some sort of storage as the data source. Then we need to convert each sample from our data into a something PyTorch can actually handle: tensors
- This bridge between our custom data (in whatever format it might be) and a standardized PyTorch tensor is the Dataset class PyTorch provides in torch.utils.data.
- we will need multiple processes to load our data, in order to assemble them into batches: tensors that encompass several samples. This is rather elaborate; but as it is also relatively generic, PyTorch readily provides all that magic in the DataLoader class. Its instances can spawn child processes to load data from a dataset in the background so that it’s ready and waiting for the training loop as soon as the loop can use it.
- At each step in the training loop, we evaluate our model on the samples we got from the data loader. We then compare the outputs of our model to the desired  output (the targets) using some criterion or loss function.
  
![ ](/images/dl2.JPG "PyTotch deep learning stages")

- The training loop might be the most unexciting yet most time-consuming part of a deep learning project. At the end of it, we are rewarded with a model whose parameters have been optimized on our task: the trained model depicted to the right of the training loop in the figure.
- PyTorch defaults to an immediate execution model (eager mode). Whenever an instruction involving PyTorch is executed by the Python interpreter, the corresponding operation is immediately carried out by the underlying C++ or CUDA implementation.

