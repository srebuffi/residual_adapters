## Parametric families of deep neural networks with residual adapters

Pytorch backbone codes for the papers:
- NIPS 2017: "Learning multiple visual domains with residual adapters", https://papers.nips.cc/paper/6654-learning-multiple-visual-domains-with-residual-adapters.pdf
- CVPR 2018: "Efficient parametrization of multi-domain deep neural networks", https://arxiv.org/pdf/1803.10082.pdf 

Page of our associated **Visual Domain Decathlon challenge** for multi-domain classification: http://www.robots.ox.ac.uk/~vgg/decathlon/

## Abstract 

A practical limitation of deep neural networks is their high degree of specialization to a single task and visual domain.
To overcome this limitation, in these papers we propose to consider instead universal parametric families of neural
networks, which still contain specialized problem-specific models, but differing only by a small number of parameters.
We study different designs for such parametrizations, including
series and parallel residual adapters. We show that, in order to maximize performance, it is necessary
to adapt both shallow and deep layers of a deep network,
but the required changes are very small. We also show that
these universal parametrization are very effective for transfer
learning, where they outperform traditional fine-tuning
techniques.

#### Code

##### Requirements
- Pytorch (at least version 3.0)
- COCO API (from https://github.com/cocodataset/cocoapi)

##### Launching the code
First download the data with ``download_data.sh /path/to/save/data/``. Please copy ``decathlon_mean_std.pickle`` to the data folder. 

To train a dataset from scratch:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_from_scratch.py --dataset cifar100 --wd3x3 1. --wd 5. --mode bn ``

To train a dataset with parallel adapters put on a pretrained 'off the shelf' deep network:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_adapters.py --dataset cifar100 --wd1x1 1. --wd 5. --mode parallel_adapters --source /path/to/net``
   
To train a dataset with series adapters put on a pretrained deep network (with adapters in it during pretraining):

``CUDA_VISIBLE_DEVICES=2 python train_new_task_adapters.py --dataset cifar100 --wd1x1 1. --wd 5. --mode series_adapters --source /path/to/net``

To train a dataset with series adapters put on a pretrained 'off the shelf' deep network:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_adapters.py --dataset cifar100 --wd1x1 1. --wd 5. --mode series_adapters --source /path/to/net``

To train a dataset with normal finetuning from a pretrained deep network:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_finetuning.py --dataset cifar100  --wd 5. --mode bn --source /path/to/net``

##### Pretrained networks
We pretrained networks on ImageNet (with reduced resolution):
- a ResNet 26 inspired from the original ResNet from [He,16]: https://drive.google.com/open?id=1y7gz_9KfjY8O4Ue3yHE7SpwA90Ua1mbR
- the same network with series adapters already in it:https://drive.google.com/open?id=1f1eBQY6eHm616SAt0UXxY9RldNM9XAHb

##### Results of the commands above with the pretrained networks
So we train on CIFAR 100 and evaluate on the eval split:

|        |     Val. Acc.     | 
| :------------ | :-------------: | 
| Scratch       |     75.23     |     
| Parallel adapters     |   80.61    |      
| Series adapters       |     80.17      |        
| Series adapters (off the shelf)       |     70.72      |     
| Normal finetuning       |     78.40      |        

## If you consider citing us

For the Visual Domain Decathlon challenge and the series adapters:


        @inproceedings{Rebuffi17,
          author       = "Rebuffi, S-A and Bilen, H. and Vedaldi, A.",
          title        = "Learning multiple visual domains with residual adapters",
          booktitle    = "Advances in Neural Information Processing Systems",
          year         = "2017",
        }


For the parallel adapters:


        @inproceedings{ rebuffi-cvpr2018,
        author = { Sylvestre-Alvise Rebuffi and Hakan Bilen and Andrea Vedaldi },
        title = {Efficient parametrization of multi-domain deep neural networks},
        booktitle = CVPR,
        year = 2018,
        }

