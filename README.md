# Deep Declarative Networks

Deep Declarative Networks (DDNs) are a class of deep learning model that allows optimization problems
to be embedded within an end-to-end learnable network. This repository maintains code and resources for
developing and understanding DDN models.

You can find more details in [this paper](https://arxiv.org/abs/1909.04866), which if you would like to
reference in your research please cite as:
```
@techreport{Gould:PrePrint2019,
  author    = {Stephen Gould and
               Richard Hartley and
               Dylan Campbell},
  title     = {Deep Declarative Networks: A New Hope},
  eprint    = {arXiv:1909.04866},
  month     = {Sep},
  year      = (2019}
}
```

## Running code

When running code from the commandline make sure you add the `ddn` package to your PYTHONPATH. For example:

```
export PYTHONPATH=${PYTHONPATH}:ddn
python tests/testBasicDeclNodes.py
```

## License

The `ddn` library is distributed under the MIT license. See the [LICENSE](LICENSE) file for details.
