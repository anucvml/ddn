# Deep Declarative Networks

Deep Declarative Networks (DDNs) are a class of deep learning model that allows for optimization problems
to be embedded within an end-to-end learnable network. This repository maintains code,
[tutorials](https://nbviewer.jupyter.org/github/anucvml/ddn/tree/master/tutorials/) and other
[resources](https://github.com/anucvml/ddn/wiki/Resources) for developing and understanding DDN models.

You can find more details in [this paper](https://arxiv.org/abs/1909.04866), which if you would like to
reference in your research please cite as:
```
@techreport{Gould:PrePrint2019,
  author      = {Stephen Gould and
                 Richard Hartley and
                 Dylan Campbell},
  title       = {Deep Declarative Networks: A New Hope},
  eprint      = {arXiv:1909.04866},
  institution = {Australian National University (arXiv:1909.04866)},
  month       = {Sep},
  year        = {2019}
}
```

## Running code

When running code from the command line make sure you add the `ddn` package to your PYTHONPATH. For example:

```
export PYTHONPATH=${PYTHONPATH}:ddn
python tests/testBasicDeclNodes.py
```

Documentation for the `ddn` package is provided in the `ddn` directory.
Interactive tutorials should be opened in Jupyter notebook:

```
cd tutorials
jupyter notebook
```

or viewed using using [jupyter.org's notebook viewer](https://nbviewer.jupyter.org/github/anucvml/ddn/tree/master/tutorials/).

Reference (PyTorch) applications for image and point cloud classification can be found under the `apps`
directory. See the `README` files therein for instructions on installation and how to run.

## License

The `ddn` library is distributed under the MIT license. See the [LICENSE](LICENSE) file for details.
