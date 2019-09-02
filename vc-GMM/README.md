# Dependencies
* Cmake
* Blaze

# How to install blaze (needs CMAKE)

1. get the source code from here: https://bitbucket.org/blaze-lib/blaze/src/master/
2. sudo apt-get install libblas-dev liblapack-dev
3. cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
4. sudo make install

# About the Clustering Algorithm

This project provides C++ source code for a very efficient clustering algorithm: *vc-GMM* [3]. In general, this novel algorithm can be applied to any metric data to which standard *k*-means or Gaussian Mixture Models (GMMs) are applied to. Especially for large-scale problems (very large numbers of data points, very large numbers of clusters, high dimensionality of data points) the algorithm is substantially faster than previous *k*-means versions (or GMM algorithms). Efficiency increases are also warranted for medium and small dimensionalities of the data points (i.e., small to intermediate numbers of features). For large-scale problems, the algorithm is on standard benchmarks several times and up to an order of magnitude faster than already very efficient recent *k*-means versions (while providing approximately equal or similar clustering results). "Large-scale" means hundreds of thousands of data points (or many more), hundreds of clusters (or many more), and tens to hundreds of features (or many more). The algorithm's output includes the estimated cluster centers and the estimated global cluster variance.
If scientific publications are produced using this code for *vc-GMM* or an implementation derived from *vc-GMM*, the paper [3] is the reference for the merged approach (and paper [1] for the variational EM part and paper [5] for the coreset part).

# Properties of the Algorithm

The efficiency of the clustering algorithm results from a combination of variational Expectation Maximization (variational EM) for GMMs and coresets for *k*-means and GMMs. The scientific details are described in the papers [1] and [3]. The main algorithm is an implementation of the algorithm *vc-GMM* described in [3]. *vc-GMM* assumes a GMM data model with equal mixing proportions, isotropic clusters, and equal cluster variances. Like *k*-means, the algorithm learns the cluster centers and the global cluster variance. *vc-GMM* is, therefore, similar to *k*-means and can in general be applied whenever *k*-means gives sufficiently good results but is too slow. However, *vc-GMM* goes beyond *k*-means by providing cluster membership estimates (responsibilities) per data point (softer clustering) and lower log-likelihood bounds. In practice, *vc-GMM* can be expected to estimate cluster centers more reliably than *k*-means especially in cases of larger cluster overlaps. *vc-GMM* will not be able to perform well, e.g., for very elliptical clusters (when standard GMM data models are preferable). *vc-GMM* also does not estimate the number of clusters like fully Bayesian GMM approaches. However, *vc-GMM* can be used in conjunction with model selection approaches (e.g. BIC) to estimate the number of clusters.

# Installation

This project requires the [blaze library](https://bitbucket.org/blaze-lib/blaze/src/master/) version 3.3.
Please follow the [installation guide](https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation) to make it available on your system.

# Details

The algorithm *vc-GMM* [3] optimizes a lower bound of the log-likelihood (free energy) of isotropic Gaussian Mixture Models (equal mixing proportions and a single variance parameter) on a weighted subset of a dataset. The subset is constructed as a lightweight coreset [5] for *k*-means.

The algorithm *var-GMM-S* [1] is reproduced by *vc-GMM* on the whole dataset with identical weights.

[1] and [3] demonstrate that *vc-GMM* and *var-GMM-S* can achieve significant speedups in comparison with *k*-means (Lloyd's algorithm) while still obtaining competitive clustering results (measured in terms of quantization error, see [2] for the relation between *k*-means and Gaussian Mixture Models).
This is realized by the definition of search spaces that allow to efficiently assign datapoints to relevant clusters without having to evaluate all N * C possible combinations for N datapoints with C clusters.
These search spaces are parameterized by two constants *C_p* and *G* and can be improved with additional random sampling by setting a boolean flag *plus1* (see [1] and [3]).
The runtime of each EM iteration scales *sublinearly* with the number of clusters (see [1] for details).

This implementation is a partially modified version of the source used to produce the results in paper [3] with support for multithreading for the E- and M-step.
Modifications include a refactoring of the seeding algorithm, details of the parallelization and bug fixes among others.

Furthermore, we have included an automatic estimation of the initial cluster variance, which can optionally be replaced by hand-set values.

# Usage

The input dataset is a [N, D] *blaze* matrix in row major order where rows represent D-dimensional feature vectors.
It can be parsed from a whitespace separated text file with the *loadtxt()* function.
Seeding is implemented via *AFK-MC²* [4], which uses Markov chains of length *chain_length* to sample initial cluster centers.
Note that results are reproducible only for runs with a fixed number of threads.

For *vc-GMM* the usage is:

```c++
#include "clustering.hxx"

int main(void) {

    blaze::DynamicMatrix<double, blaze::rowMajor> x;
    loadtxt("<path-to-dataset>.txt", x); // read dataset from whitespace separated text file

    int C_p = 5;    // 1 <= C_p <= C
    int G   = 5;    // 2 <= G   <= C
    bool plus1      = true;

    int nthreads = std::thread::hardware_concurrency(); // number of C++11 threads
    int seed     = 123;                                 // seed for random number generator

    int N_core        = ...; // size of subset (lightweight coreset)
    int C             = ...; // number of cluster centers
    int chain_length  = ...; // chain length for AFK-MC² seeding

    clustering<double> vc_GMM(x, N_core, C, chain_length, C_p, G, plus1, nthreads, seed);

    vc_GMM.converge(1e-4); // computes EM-iterations until the relative change in free energy falls below 1e-4

    savetxt("<path-to-result>.txt", vc_GMM.cluster_centers()); // write the cluster centers, i.e. means of the GMM
}
```

For *var-GMM-S* simply omit the *N_core* parameter:

```c++
...
clustering<double> var_GMM_S(x, C, chain_length, C_p, G, plus1, nthreads, seed);
...
```

Additionally the following methods are available:

```c++
vc_GMM.estep() // computes a single E-step
vc_GMM.mstep() // computes a single M-step

vc_GMM.set_variance(value) // manually set the variance parameter to value, e.g. for initialization
vc_GMM.lower_bound() // returns the lower bound of the log-likelihood (the lower bound is typically referred to as the "free energy" or "ELBO")"
vc_GMM.variance() // returns the current variance
vc_GMM.error() // returns the quantization error
```

Compilation requires a C++14 compiler and the *blaze* library (see [installation guide]( https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation)).
In the simplest case, copy the *blaze* subdirectory from [blaze-3.3.tar.gz](https://bitbucket.org/blaze-lib/blaze/downloads/) (or any later version) into the same folder that contains the project's source files.
Then, e.g. on Ubuntu 16.04 LTS with gcc version 5.4.0, compile with

```
g++ -DNDEBUG -std=c++14 -I. -O2 -o program example.cxx -lpthread
```

To run the program, make sure to replace *<path-to-dataset>.txt*, set the values of the free parameters accordingly, recompile, and run

```
./program
```

For better performance it is recommended to use [blaze archives](https://bitbucket.org/blaze-lib/blaze/wiki/Serialization) instead of reading and writing text files with *loadtxt()* and *savetxt()*.
Also consider enabling SIMD vectorization with e.g. *-mavx*.

# License

This project is provided "as is" under the Academic Free License (AFL) v3.0

# References

1. D. Forster and J. Lücke. Can clustering scale sublinearly with its clusters? A variational EM acceleration of GMMs and *k*-means. In Proceedings *AISTATS*, pages 124–132, 2018.
2. J. Lücke and D. Forster. k-means as a variational EM approximation of Gaussian mixture models. *arXiv preprint arXiv:1704.04812*, 2017.
3. F. Hirschberger, D. Forster and J. Lücke. Large Scale Clustering with Variational EM for Gaussian Mixture Models. *arXiv preprint arXiv:1810.00803*, 2019.
4. O. Bachem, M. Lucic, H. Hassani, and A. Krause. Fast and provably good seedings for k-means. In *Proc. Advances in Neural Information Processing Systems*, pages 55–63, 2016a.
5. O. Bachem, M. Lucic, and A. Krause. Scalable k-means clustering via lightweight coresets. In *Proceedings KDD*, pages 1119–1127, 2018.
