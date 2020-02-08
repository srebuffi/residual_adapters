## Backbone codes for the parallel residual adapters with MatConvNet

The ``cnn_cifar.m`` adds parallel residual adapters to a ResNet50 network pretrained on ImageNet and trains on CIFAR10/100. To adapt the architecture to the CIFAR input size, there are two possible options for ``opts.modelType``: 'new_conv1' for replacing the original conv1 by a 3x3 conv1 layer (which will be trained with the adapters) or 'reduce_stride' where the original parameters of the conv1 are preserved and frozen but the stride is reduced to 1. In both cases, the first maxpool layer is deleted.
