# Towards Understanding Gradient Approximation in Equality Constrained Deep Declarative Networks

Code to reproduce experiments appearing in the [ICML 2023 Workshop on Differentiable Anything](https://differentiable.xyz/) paper "Towards Understanding Gradient Approximation in Equality Constrained Deep Declarative Networks."

If you would like to reference this paper or code in your research please cite as:

```
@inproceedings{Gould:ICML23w,
  author    = {Stephen Gould and
               Ming Xu and
               Zhiwei Xu and
               Yanbin Liu},
  title     = {Towards Understanding Gradient Approximation in Equality Constrained Deep Declarative Networks},
  booktitle = {ICML Workshop on Differentiable Almost Everything: Differentiable Relaxations, Algorithms, Operators, and Simulators},
  year      = {2023}
}
```

## Running code

When running code from the command line make sure you add the `ddn` package to your PYTHONPATH. For example:

```
export PYTHONPATH=${PYTHONPATH}:ddn
python descent.py
```
