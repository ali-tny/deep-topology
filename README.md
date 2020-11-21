Topological Deep Learning
=========================

Tensorflow implementation of papers that use persistent homology features:
- [Topological Autoencoders](https://arxiv.org/pdf/1906.00722.pdf) inspired by the authors' own 
  [PyTorch implementation](https://github.com/BorgwardtLab/topological-autoencoders)
- [Topologically Densified Distributions](https://arxiv.org/pdf/2002.04805.pdf)

In both cases, the interesting part is the calculation of 0D homology features (ie edges in the
minimum spanning tree). This implementation includes a numpy implementation that runs on the CPU,
and a pure Tensorflow tf.function implementation that runs in graph execution mode.

References
----------
**Topological Autoencoders**, Michael Moor, Max Horn, Bastian Rieck, Karsten Borgwardt [(ICML 2020)](https://arxiv.org/pdf/1906.00722.pdf)

**Topologically Densified Distributions**, Christoph Hofer, Florian Graf, Marc Niethammer, Roland Kwitt [(ICML 2020)](https://arxiv.org/abs/2002.04805.pdf)

**Connectivity-Optimized Representation Learning via Persistent Homology**, Christoph Hofer, Roland Kwitt, Marc Niethammer, Mandar Dixit [(PMLR 97:2751-2760, 2019)](http://proceedings.mlr.press/v97/hofer19a.html)
