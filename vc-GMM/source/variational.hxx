/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#ifndef VARIATIONAL
#define VARIATIONAL

#define _USE_MATH_DEFINES
#include <cmath>

#include <stdexcept>
#include <random>
#include <vector>
#include <array>
#include <limits>

#include "blaze/Blaze.h"
#include "threads.hxx"

template <typename T>
struct variational {
	struct _tab {
		size_t opc;
		size_t ref;
        _tab(void)
            : opc(std::numeric_limits<size_t>::max())
            , ref(std::numeric_limits<size_t>::max())
        {}
	};
	struct _dcn {
		size_t clu;
		T      key;
	};
	struct _sel {
		size_t clu;
		size_t num;
		T      key;
	};
    const blaze::DynamicMatrix<T, blaze::rowMajor>& data;
    const blaze::DynamicVector<T>& weight;
	      blaze::DynamicMatrix<T, blaze::rowMajor>& mean;

    size_t N;
    size_t C;
    size_t D;
    size_t K;	// equal to C_p
    size_t G;

    blaze::DynamicMatrix<_dcn,   blaze::rowMajor> Qn;
    blaze::DynamicMatrix<_dcn,   blaze::rowMajor> Dc;
    blaze::DynamicMatrix<size_t, blaze::rowMajor> Gc;
    blaze::DynamicVector<size_t> Ps;
    blaze::DynamicVector<size_t> Op;

    std::vector<std::array<std::mt19937_64, 64>> mt;
	tp threads;

	bool p1;

	T quantization;
    T free_energy;
	T variance;
	T norm;

	void init_Kn(void);
	void init_Gc(void);
	size_t expectation(void);
	void estimate(void);
	T guess_variance(void);

	variational(void) {};
    variational(
 		const blaze::DynamicMatrix<T, blaze::rowMajor>& data,
        const blaze::DynamicVector<T>& weight,
              blaze::DynamicMatrix<T, blaze::rowMajor>& mean,
        size_t K,
        size_t G,
        bool p1,
		T variance,
        size_t nthreads,
		size_t seed
    );

    size_t estep(void);
    void fit(void);
};

template <typename T>
T
variational<T>::guess_variance(void)
{
	(*this).variance = (*this).quantization / ((*this).norm * D);
	return (*this).variance;
}

template <typename T>
size_t
variational<T>::estep(void)
{
	size_t count = expectation();
    estimate();

	return count;
}

template <typename T>
variational<T>::variational(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& data,
    const blaze::DynamicVector<T>& weight,
          blaze::DynamicMatrix<T, blaze::rowMajor>& mean,
    size_t K,
    size_t G,
    bool p1,
    T variance,
	size_t nthreads,
	size_t seed)
    : data(data)
    , weight(weight)
    , mean(mean)
    , N(data.rows())
    , C(mean.rows())
    , D(mean.columns())
    , K(K)
    , G(G)
    , Qn(N, K)
    , Dc(N, std::min(K * G, C) + p1)
    , Gc(C, G)
    , Ps(N)
    , Op(N)
    , mt(nthreads)
	, threads(nthreads)
    , p1(p1)
    , variance(variance)
{
	for (std::size_t t = 0; t < threads.size(); t++) {
		std::seed_seq seq{seed + t};
		mt[t][0].seed(seq);
		mt[t][0].discard(1e3);
	}

    if (data.rows() != weight.size()) {
         throw std::invalid_argument("data.rows() != weight.size()");
    }
    if (data.columns() != mean.columns()) {
         throw std::invalid_argument("data.columns() != mean.columns()");
    }
	if ((K == 0) || (K > C)) {
		throw std::invalid_argument("invalid parameter C_p");
	}
	if ((G < 2) || (K > C)) {
		throw std::invalid_argument("invalid parameter G");
	}
	if (!(variance > 0)) {
		throw std::invalid_argument("invalid variance");
	}
	if (nthreads == 0) {
		throw std::invalid_argument("invalid number of threads");
	}
	if (N == 0) {
		throw std::invalid_argument("number of data points is zero");
	}
	if (C == 0) {
		throw std::invalid_argument("number of cluster centers is zero");
	}
	if (D == 0) {
		throw std::invalid_argument("number of features is zero");
	}

    init_Kn();
    init_Gc();

	norm = 0;
	for (size_t n = 0; n < N; n++) {
		norm += weight[n];
	}
}


template <typename T>
void variational<T>::init_Kn(void)
{
    using blaze::row;

    auto tab = std::vector<blaze::DynamicVector<_tab>>(
		threads.size(),
		blaze::DynamicVector<_tab>(C)
	);

    threads.parallel(N, [&] (size_t n, size_t t)
	-> void {
		std::uniform_int_distribution<size_t> uniform_C(0, C - 1);
		for (auto& it : row(Qn, n)) {
			size_t u;
			do {
				u = uniform_C(mt[t][0]);
			} while (tab[t][u].opc == n);

            tab[t][u].opc = n;
			it.clu = u;
		}
	});
}


template <typename T>
void variational<T>::init_Gc(void)
{
    using blaze::subvector;
    using blaze::row;

	auto tab = std::vector<blaze::DynamicVector<_tab>>(
		threads.size(),
		blaze::DynamicVector<_tab>(C)
	);

	threads.parallel(C, [&] (size_t c, size_t t)
	-> void {
		std::uniform_int_distribution<size_t> uniform_C(0, C - 1);
		Gc(c, 0) = c;
        tab[t][c].opc = c;
		for (auto& it : subvector(row(Gc, c), 1, G - 1)) {
			size_t u;
			do {
				u = uniform_C(mt[t][0]);
			} while (tab[t][u].opc == c);

            tab[t][u].opc = c;
			it = u;
		}
	});
}

template <typename T>
void
variational<T>::estimate(void)
{
    auto tab = std::vector<blaze::DynamicVector<_tab>>(
		threads.size(),
		blaze::DynamicVector<_tab>(C)
	);
    auto sel = std::vector<blaze::DynamicVector<_sel>>(
		threads.size(),
		blaze::DynamicVector<_sel>(C)
	);

    auto inverse = std::vector<std::vector<size_t>>(
        C,
        std::vector<size_t>{}
    );
	for (size_t n = 0; n < N; n++) {
		inverse[Op[n]].push_back(n);
	}

	threads.parallel(C, [&] (size_t c, size_t t)
	-> void {
		size_t inc = 0;
		for (auto& n : inverse[c]) {
			for (auto& it : subvector(row(Dc, n), 0, Ps[n])) {

				size_t u  = it.clu;
				T      d2 = it.key;

				if (tab[t][u].opc != c) {
					tab[t][u].opc  = c;
					tab[t][u].ref  = inc;

					sel[t][inc].clu = u;
					sel[t][inc].num = 1;
					sel[t][inc].key = std::sqrt(d2);
					inc++;
				} else {
					sel[t][tab[t][u].ref].num += 1;
					sel[t][tab[t][u].ref].key += std::sqrt(d2);
				}
			}
		}

        if (inc) {
            for (size_t i = 0; i < inc; i++) {
                if (sel[t][i].clu != c) {
                    sel[t][i].key *= T(1.0)/sel[t][i].num;
                } else {
                    sel[t][i].key = T(-1);
                }
            }

            std::nth_element(
                sel[t].begin(),
                sel[t].begin() + G,
                sel[t].begin() + inc,
                [&] (auto& lhs, auto& rhs)
                -> bool {
                    return lhs.key < rhs.key;
                }
            );

            for (size_t g = 0; g < G; g++) {
                Gc(c, g) = sel[t][g].clu;
            }
        }
	});
}

template <typename T>
size_t
variational<T>::expectation(void)
{
    using blaze::row;

    auto tab = std::vector<blaze::DynamicVector<_tab>>(
		threads.size(),
		blaze::DynamicVector<_tab>(C)
	);
	auto acc = std::vector<std::array<T, 64>>(
		threads.size(),
		std::array<T, 64>{}
	);
	auto err = std::vector<std::array<T, 64>>(
		threads.size(),
		std::array<T, 64>{}
	);

    threads.parallel(N, [&] (size_t n, size_t t)
	-> void {
		size_t count = 0;
        for (auto& k : row(Qn, n)) {
            for (auto& c : row(Gc, k.clu)) {
                if (tab[t][c].opc != n) {
                    tab[t][c].opc  = n;
                    Dc(n, count).clu = c;
                    count++;
                }
            }
        }
        if (p1) {
            std::uniform_int_distribution<size_t> uniform_C(0, C - 1);
            size_t c = uniform_C(mt[t][0]);

            if (tab[t][c].opc != n) {
                tab[t][c].opc  = n;
                Dc(n, count).clu = c;
                count++;
            }
        }
		Ps[n] = count;

        for (auto& it : subvector(row(Dc, n), 0, Ps[n])) {
            it.key = sqrNorm(row(data, n) - row(mean, it.clu));
        }

		std::nth_element(
			row(Dc, n).begin(),
			row(Dc, n).begin() + K,
			row(Dc, n).begin() + Ps[n],
			[&] (auto& lhs, auto& rhs)
			-> bool {
				return lhs.key < rhs.key;
			}
		);

		_dcn tmp{
			std::numeric_limits<size_t>::max(),
			std::numeric_limits<T     >::max()
		};
		for (size_t k = 0; k < K; k++) {
            Qn(n, k) = Dc(n, k);

			if (tmp.key > Dc(n, k).key) {
				tmp     = Dc(n, k);
			}
		}
		Op[n] = tmp.clu;
		err[t][0] += weight[n] * tmp.key;

		T sum = 0;
		T lim = -std::numeric_limits<T>::max();

		for (auto& q : row(Qn, n)) {
			q.key *= -T(0.5) / variance;
		}
		for (auto& q : row(Qn, n)) {
			lim = std::max(lim, q.key);
		}
		for (auto& q : row(Qn, n)) {
			q.key = std::exp(q.key - lim);
		}
		for (auto& q : row(Qn, n)) {
			sum += q.key;
		}
		for (auto& q : row(Qn, n)) {
			q.key *= T(1.0)/sum;
		}

        acc[t][0] += weight[n] * (T(-0.5) * D * std::log(T(2.0 * M_PI) * variance));
		acc[t][0] += weight[n] * (std::log(sum) + lim);
        acc[t][0] -= weight[n] * (std::log(C));
	});

	(*this).quantization = 0;
	(*this).free_energy = 0;
	for (size_t t = 0; t < threads.size(); t++) {
		(*this).quantization += err[t][0];
		(*this).free_energy += acc[t][0];
	}

	size_t tmp_sum = 0;
	for (size_t n = 0; n < N; n++) {
		tmp_sum += Ps[n];
	}
	return tmp_sum;
}

template <typename T>
void
variational<T>::fit(void)
{
    using blaze::sqrNorm;
    using blaze::row;

    std::vector<blaze::DynamicVector<T>> _norm(
		threads.size(),
		blaze::DynamicVector<T>(C, T(0))
	);
	std::vector<blaze::DynamicMatrix<T>> _mean(
		threads.size(),
		blaze::DynamicMatrix<T>(C, D, T(0))
	);
	std::vector<std::array<T, 64>> _variance(
		threads.size(),
		std::array<T, 64>{}
	);

    threads.parallel(N, [&] (size_t n, size_t t)
	-> void {
		for (size_t k = 0; k < K; k++) {

			size_t c = Qn(n, k).clu;
			T      w = Qn(n, k).key * weight[n];

            row(_mean[t], c) += w * row(data, n);
			_norm[t][c]      += w;
		}
	});
	for (size_t t = 1; t < threads.size(); t++) {
		_mean[0] += _mean[t];
	}
	for (size_t t = 1; t < threads.size(); t++) {
		_norm[0] += _norm[t];
	}

    T sum = T(0);
    for (size_t c = 0; c < C; c++) {
        T tmp = T(1.0) / _norm[0][c];
        if (std::isfinite(tmp)) {
            row(_mean[0], c) *= tmp;
        }
        sum += _norm[0][c];
    }

	(*this).mean = _mean[0];

    threads.parallel(N, [&] (size_t n, size_t t)
	-> void {
		for (size_t k = 0; k < K; k++) {

			size_t c = Qn(n, k).clu;
			T      w = Qn(n, k).key * weight[n];

			_variance[t][0] += w * sqrNorm(row(mean, c) - row(data, n));
		}
	});
	for (size_t t = 1; t < threads.size(); t++) {
		_variance[0][0] += _variance[t][0];
	}

    (*this).variance = _variance[0][0] / (sum * D);
}

#endif
