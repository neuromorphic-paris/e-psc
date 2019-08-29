/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

// Here we document the implementation of k-means++. The function d2(...)
// provides the seeding step and kmeans(...) the optimization procedure.

#include "blaze/Blaze.h"
#include <iostream>
#include <random>
#include <cmath>

template <typename T>
void
d2(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x,      // data set
    const blaze::DynamicVector<T>& w,                       // weights
          blaze::DynamicMatrix<T, blaze::rowMajor>& m,      // cluster center
    std::mt19937_64& mt,
    size_t C                                                // cluster numbers
);

template <typename T>
void
kmeans(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x,      // data set
    const blaze::DynamicVector<T>& w,                       // weights
          blaze::DynamicMatrix<T, blaze::rowMajor>& m,      // cluster center
    std::mt19937_64& mt,
    T eps,                                                   // convergence
    T init_variance                                          // variance guess
);

/******************************************************************************/

template <typename T>
void
d2(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x,
    const blaze::DynamicVector<T>& w,
          blaze::DynamicMatrix<T, blaze::rowMajor>& m,
    std::mt19937_64& mt,
    size_t C
) {
    using blaze::sqrNorm;
    using blaze::row;

    size_t N = x.rows();
    size_t D = x.columns();

    blaze::DynamicVector<T     > min_dst(N);
    blaze::DynamicVector<size_t> min_idx(N);

    {
        std::discrete_distribution<size_t> i(w.begin(), w.end());
        m.resize(C, D);
        row(m, 0) = row(x, i(mt));

        for (size_t n = 0; n < N; n++) {
            min_dst[n] = w[n] * sqrNorm(row(x, n) - row(m, 0));
            min_idx[n] = 0;
        }
    }

    for (size_t c = 1; c < C; c++) {

        std::discrete_distribution<size_t> q(min_dst.begin(), min_dst.end());
        size_t clu = q(mt);
        row(m, c) = row(x, clu);

        for (size_t n = 0; n < N; n++) {
            T tmp = w[n] * sqrNorm(row(x, n) - row(m, c));

            if (min_dst[n] > tmp) {
                min_dst[n] = tmp;
                min_idx[n] = clu;
            }
        }
    }

}

template <typename T>
T
kmeans_partition(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x,
    const blaze::DynamicVector<T>& w,
    const blaze::DynamicMatrix<T, blaze::rowMajor>& m,
          blaze::DynamicVector<size_t>& sc,
    T variance
) {
    using blaze::sqrNorm;
    using blaze::row;

    size_t N = x.rows();
    size_t C = m.rows();
    size_t D = m.columns();

    T sum  = 0;
    T nsum = 0;
    for (size_t n = 0; n < N; n++) {
        T opt = std::numeric_limits<T>::max();
        size_t idx = std::numeric_limits<size_t>::max();
        for (size_t c = 0; c < C; c++) {
            T tmp = sqrNorm(row(x, n) - row(m, c));
            if (opt > tmp) {
                opt = tmp;
                idx = c;
            }
        }
        sc[n] = idx;
        sum += w[n] * opt / (T(2.0) * variance);
        nsum += w[n];
    }

    return - (nsum * (std::log(C) + (D / 2.0) * std::log(2.0 * M_PI * variance)) + sum);
}

template <typename T>
void
kmeans_mean(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x,
    const blaze::DynamicVector<T>& w,
          blaze::DynamicMatrix<T, blaze::rowMajor>& m,
    const blaze::DynamicVector<size_t>& sc
) {
    using blaze::swap;
    using blaze::row;

    size_t N = x.rows();
    size_t C = m.rows();

    blaze::DynamicVector<T> sum(C, T(0));
    m = 0;

    for (size_t n = 0; n < N; n++) {
        row(m, sc[n]) += w[n] * row(x, n);
        sum[sc[n]] += w[n];
    }
    for (size_t c = 0; c < C; c++) {
        if (sum[c] > 0) {
            row(m, c) /= sum[c];
        }
    }
}

template <typename T>
void
kmeans_variance(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x,
    const blaze::DynamicVector<T>& w,
    const blaze::DynamicMatrix<T, blaze::rowMajor>& m,
    const blaze::DynamicVector<size_t>& sc,
    T& variance
) {
    using blaze::sqrNorm;
    using blaze::row;

    size_t N = x.rows();
    size_t D = m.columns();

    variance = 0;

    T sum = 0;
    for (size_t n = 0; n < N; n++) {
        variance += w[n] * sqrNorm(row(x, n) - row(m, sc[n]));
        sum += w[n];
    }
    variance /= sum * D;
}

template <typename T>
void
kmeans(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x,
    const blaze::DynamicVector<T>& w,
          blaze::DynamicMatrix<T, blaze::rowMajor>& m,
    std::mt19937_64& mt,
    T eps,
    T init_variance
) {
    T bound;
    T previous;

    // k-means can be described as a variational EM approximation of Gaussian
    // mixture models (with equal mixing proportions and a single variance
    // parameter). With the inclusion of a variance parameter, we can compute
    // a lower bound to the likelihood (as for Gaussian mixture models) and
    // declare convergence based on the relative change of this lower bound.
    // For a fixed number of iterations, the variance does not affect the
    // final cluster centers

    T variance = init_variance;
    size_t iters = 0;

    blaze::DynamicVector<size_t> sc(x.rows());

    while (true) {

        bound = kmeans_partition(x, w, m, sc, variance);
        kmeans_mean(x, w, m, sc);
        kmeans_variance(x, w, m, sc, variance);
        iters++;

        std::cout << "iteration   : " << iters << std::endl;
        std::cout << "lower bound : " << bound << std::endl;

        if (iters > 1) {
            if (std::abs((bound - previous) / bound) < eps) {
                break;
            }
        }
        previous = bound;
    }
}
