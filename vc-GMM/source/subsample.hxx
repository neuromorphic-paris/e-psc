/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

// This source file implements lightweight coreset construction.
// Reference:
// O. Bachem, M. Lucic, and A. Krause. Scalable k-means clustering via
// lightweight coresets. In Proceedings KDD, pages 1119–1127, 2018.

#ifndef LWCS
#define LWCS

#include <random>
#include "blaze/Blaze.h"

// implements lightweight coreset construction for k-means
template <typename T>
void
lwcs(
	const blaze::DynamicMatrix<T, blaze::rowMajor>& x,	// dataset
	      blaze::DynamicMatrix<T, blaze::rowMajor>& c,	// coreset
	      blaze::DynamicVector<T>& w,					// weights
	size_t N_core,
    std::mt19937_64& mt									// random number generator
) {
    using blaze::sqrNorm;
    using blaze::row;

	blaze::DynamicVector<T, blaze::rowVector> u(x.columns(), T(0));
	blaze::DynamicVector<T>                   q(x.rows());

	size_t N = x.rows();

	c.resize(N_core, x.columns());
	w.resize(N_core);

	// compute mean
	for (size_t n = 0; n < N; n++) {
		u += row(x, n);
	}
	u *= T(1.0)/N;

	// compute proposal distribution
	for (size_t n = 0; n < N; n++) {
		q[n] = sqrNorm(row(x, n) - u);
	}
	T sum = T(0);
	for (size_t n = 0; n < N; n++) {
		sum += q[n];
	}
	for (size_t n = 0; n < N; n++) {
		q[n] = T(0.5) * (q[n] / sum + T(1.0) / N);
	}

	// sample coreset and set weights
	std::discrete_distribution<size_t> dst(q.begin(), q.end());
	for (size_t m = 0; m < N_core; m++) {
		size_t n = dst(mt);
		row(c, m) = row(x, n);
		w[m] = T(1.0) / (q[n] * N_core);
	}
}

#endif
