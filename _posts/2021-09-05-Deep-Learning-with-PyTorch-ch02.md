---
toc: true
layout: post
description: A synopsys of chapter 02 of book "Deep Learning with PyTorch".
categories: [markdown]
title: Chapter 02 summary
---
# Playning with Pretrained models in PyTorch
## What is a pretrained model?
- We can think of a pretrained neural network as similar to a program that takes inputs and generates outputs. The behavior of such a program is dictated by the architecture of the neural network and by the examples it saw during training, in terms of desired input-output pairs, or desired properties that the output should satisfy.
  
We will explore three popular pretrained models: a model that can
label an image according to its content, another that can fabricate a new image from a real image, and a model that can describe the content of an image using proper English sentences.

The scope of this chapter is only how to run a pretrained model using PyTorch is a useful skill—full stop. Basically We are going to take our own images and feed them into our pretrained model, as described in figure below. This will result in a list of predicted labels for that image, which we can then examine to see what the model thinks our image is. Some images will have predictions that are accurate, and others will not!
![Pretrained ](/images/dl3.JPG)