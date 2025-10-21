# Practical Kernel Selection for Kernel-based Conditional Independence Test

This repository provides an implementation for the automatic selection of kernel parameters in the Kernel-based Conditional Independence (KCI) test.
Given a predefined list of candidate kernel bandwidths, the method computes the estimated test power for all candidates in parallel and selects the kernels with the highest estimated power for the final conditional independence testing.
For more details, please refer to “Practical Kernel Selection for Kernel-based Conditional Independence Test” (NeurIPS 2025)
.
Dependencies: `pytorch 2.3.1, joblib 1.4.2, scipy~=1.14.1, scikit-learn~=1.5.2`


## Run
First, run the requirements with `pip install -r requirements.txt`.

You can run `python Main.sh` to test the codes. 

## Acknowledgments
- Our implementation is highly based on the KCI implementation in causal discovery python package [pip link](https://github.com/py-why/causal-learn) and [ducoment link](https://causal-learn.readthedocs.io/en/latest/).

If you find it useful, please consider citing: 
```bibtex
@inproceedings{
wang2024optimal,
title={Optimal Kernel choice for score function-based causal discovery},
author={Wenjie Wang, Biwei Huang, Feng Liu, Xinge You, Tongliang Liu, Kun Zhang, Mingming Gong},
booktitle={International conference on machine learning},
year={2024},
organization={PMLR}
}
```

```bibtex
@inproceedings{zhang2011kernel,
  title={Kernel-based conditional independence test and application in causal discovery},
  author={Zhang, Kun and Peters, Jonas and Janzing, Dominik and Sch{\"o}lkopf, Bernhard},
  booktitle={Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence},
  pages={804--813},
  year={2011}
}
```
