/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#ifndef UTILITY
#define UTILITY

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "blaze/Blaze.h"
#include "threads.hpp"

// computes the quantization error
// x - dataset
// s - cluster centers
template <typename T>
T quantization(const blaze::DynamicMatrix<T, blaze::rowMajor>& x, const blaze::DynamicMatrix<T, blaze::rowMajor>& s, tp& threads) {

    size_t N = x.rows();
    size_t C = s.rows();

    std::vector<std::array<T, 64>> sum(threads.size());
    threads.parallel(N, [&] (size_t n, size_t t) -> void {
        T d2 = std::numeric_limits<T>::max();
        for (size_t c = 0; c < C; c++) {
            d2 = std::min(d2, blaze::sqrNorm(blaze::row(x, n) - blaze::row(s, c)));
        }
        sum[t][0] += d2;
    });

    T res = T(0);
    for (auto& it : sum) {
        res += it[0];
    }

    return res;
}

// reads a blaze matrix from whitespace separated text file
template <typename T>
void loadtxt(const std::string& path, blaze::DynamicMatrix<T, blaze::rowMajor>& x) {
    std::cout << "reading file ";
    std::cout << path;
    std::cout << "... ";
    std::cout << std::flush;

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::invalid_argument("could not open dataset file");
    }
    std::string line;
    std::vector<std::vector<T>> buf;

    while (std::getline(ifs, line)) {
        std::istringstream iss{line};
        buf.push_back(std::vector<T>{});
        T el;
        while (iss >> el) {
            buf.back().push_back(el);
        }
        if (buf.back().size() != buf.front().size()) {
            throw std::invalid_argument("different row sizes");
        }
    }
    if (!buf.empty()) {

        size_t row = buf.size();
        size_t col = buf.front().size();

        x.resize(row, col);

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                x(i, j) = buf[i][j];
            }
        }
    }

    std::cout << "done!\n";
    std::cout << "data points N = " << x.rows()    << "\n";
    std::cout << "features    D = " << x.columns() << "\n";
    std::cout << std::endl;
}

// writes blaze matrix as a text file
template <typename T>
void savetxt(const std::string& path, const blaze::DynamicMatrix<T>& x) {
    std::ofstream ofs(path);

    for (size_t i = 0; i < x.rows(); i++) {
        for (size_t j = 0; j < x.columns(); j++) {
            ofs << x(i, j) << "\t";
        }
        ofs << "\n";
    }
}

#endif
