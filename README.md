# Deep Declarative Networks

Deep Declarative Networks (DDNs) are a class of deep learning model that allows for optimization problems
to be embedded within an end-to-end learnable network. This repository maintains code,
[tutorials](https://nbviewer.jupyter.org/github/anucvml/ddn/tree/master/tutorials/) and other
[resources](https://github.com/anucvml/ddn/wiki/Resources) for developing and understanding DDN models.

You can find more details in [this paper](https://arxiv.org/abs/1909.04866) (also [here](https://ieeexplore.ieee.org/document/9355027)),
which if you would like to reference in your research please cite as:
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

or

```
@journal{Gould:PAMI2022,
  author      = {Stephen Gould and
                 Richard Hartley and
                 Dylan Campbell},
  title       = {Deep Declarative Networks},
  journal     = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year        = {2022},
  month       = {Aug},
  volume      = {44},
  pages       = {3988--4004},
  doi         = {10.1109/TPAMI.2021.3059462}
}
```

## Running code

When running code from the command line make sure you add the `ddn` package to your PYTHONPATH. For example:

```
export PYTHONPATH=${PYTHONPATH}:ddn
python tests/testBasicDeclNodes.py
```

Documentation for the `ddn` package is provided in the `ddn` directory and many examples given in the interactive `turotials`.
These should be opened in Jupyter notebook:

```
cd tutorials
jupyter notebook
```

or viewed using using [jupyter.org's notebook viewer](https://nbviewer.jupyter.org/github/anucvml/ddn/tree/master/tutorials/).

Reference (PyTorch) applications for image and point cloud classification can be found under the `apps`
directory. See the `README` files therein for instructions on installation and how to run.

## License

The `ddn` library is distributed under the MIT license. See the [LICENSE](LICENSE) file for details.
