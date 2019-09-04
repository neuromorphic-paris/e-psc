/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

// This source file implements AFK-MC^2 seeding.
// Reference:
// O. Bachem, M. Lucic, H. Hassani, and A. Krause. Fast and provably good
// seedings for k-means. In Proc. Advances in Neural Information Processing
// Systems, pages 55–63, 2016a.

#ifndef AFKMC2_SEEDING
#define AFKMC2_SEEDING

#include <random>
#include "blaze/Blaze.h"

// x     - dataset or coreset
// w     - weights
// s     - cluster centers
// mt    - random number generator
// C     - number of cluster centers
// chain - Markov chain length
template <typename T>
void afkmc2(const blaze::DynamicMatrix<T, blaze::rowMajor>& x, const blaze::DynamicVector<T> w, blaze::DynamicMatrix<T, blaze::rowMajor>& s, std::mt19937_64& mt, size_t C, size_t chain) {
    using blaze::sqrNorm;
    using blaze::row;

    size_t N = x.rows();
    size_t D = x.columns();

    // draw first cluster
    {
        std::discrete_distribution<size_t> i(w.begin(), w.end());
        s.resize(C, D);
        row(s, 0) = row(x, i(mt));
    }

    // compute proposal distribution
    blaze::DynamicVector<T> q(N);
    {
        for (size_t n = 0; n < N; n++) {
            q[n] = sqrNorm(row(x, n) - row(s, 0)) * w[n];
        }
        T dsum = T(0);
        T wsum = T(0);
        for (size_t n = 0; n < N; n++) {
            dsum += q[n];
            wsum += w[n];
        }
        for (size_t n = 0; n < N; n++) {
            q[n] = T(0.5) * (q[n] / dsum + w[n] / wsum);
        }
    }

    std::discrete_distribution<size_t> draw_q(q.begin(), q.end());
    std::uniform_real_distribution<T> uniform(T(0.0), T(1.0));

    for (size_t c = 1; c < C; c++) {

        // initialize a new Markov chain
        size_t x_idx = draw_q(mt);
        T x_key;

        // compute distance to closest cluster
        T dist = std::numeric_limits<T>::max();
        for (size_t _c = 0; _c < c; _c++) {
            dist = std::min(dist, sqrNorm(row(x, x_idx) - row(s, _c)));
        }
        x_key = dist * w[x_idx];

        // Markov chain
        for (size_t i = 1; i < chain; i++) {

            // draw new potential cluster center from proposal distribution
            size_t y_idx = draw_q(mt);
            T y_key;

            // compute distance to closest cluster
            T dist = std::numeric_limits<T>::max();
            for (size_t _c = 0; _c < c; _c++) {
                dist = std::min(dist, sqrNorm(row(x, y_idx) - row(s, _c)));
            }
            y_key = dist * w[y_idx];


            // determine the probability to accept the new sample y_idx
            T y_prob = y_key / q[y_idx];
            T x_prob = x_key / q[x_idx];

            if (((y_prob / x_prob) > uniform(mt)) || (x_prob == 0)) {
                x_idx = y_idx;
                x_key = y_key;
            }
        }

        row(s, c) = row(x, x_idx);
    }
}

#endif
