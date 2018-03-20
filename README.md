# Parametric families of deep neural networks with residual adapters

Pytorch backbone codes for the papers:
- NIPS 2017: "Learning multiple visual domains with residual adapters", https://papers.nips.cc/paper/6654-learning-multiple-visual-domains-with-residual-adapters.pdf
- CVPR 2018: "Efficient parametrization of multi-domain deep neural networks", 

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

## If you consider citing us

    @inproceedings{Rebuffi17,
      author       = "Rebuffi, S-A and Bilen, H. and Vedaldi, A.",
      title        = "Learning multiple visual domains with residual adapters",
      booktitle    = "Advances in Neural Information Processing Systems",
      year         = "2017",
    }

    @inproceedings{ rebuffi-cvpr2018,
       author = { Sylvestre-Alvise Rebuffi and Hakan Bilen and Andrea Vedaldi },
       title = {Efficient parametrization of multi-domain deep neural networks},
       booktitle = CVPR,
       year = 2018,
    }
