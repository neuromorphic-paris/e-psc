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
#include <string>

#include "threads.hpp"

// external dependency
#include "blaze/Blaze.h"
#include "cnpy.h"

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

    // inference - which c has the lowest d (argmin)
    // F
    // histogram
    
    T res = T(0);
    for (auto& it : sum) {
        res += it[0];
    }

    return res;
}

// infers the cluster of a data point
// x - dataset
// s - cluster centers
template <typename T>
std::vector<size_t> inference(const blaze::DynamicMatrix<T, blaze::rowMajor>& x, const blaze::DynamicMatrix<T, blaze::rowMajor>& s, tp& threads) {
    size_t N = x.rows();
    size_t C = s.rows();
    
    std::vector<size_t> assigned_clusters(x.rows(), 0);
    threads.parallel(N, [&] (size_t n, size_t t) -> void {
        T d2 = std::numeric_limits<T>::max();
        size_t c_min = std::numeric_limits<size_t>::max();
        
        // find the cluster with the lowest quantization error
        for (size_t c = 0; c < C; c++) {
            T tmp_d2 = blaze::sqrNorm(blaze::row(x, n) - blaze::row(s, c));
            if (tmp_d2 < d2) {
                d2 = tmp_d2;
                c_min = c;
            }
        }
        assigned_clusters[n] = c_min;
    });
    
    return assigned_clusters;
}

// reads a blaze matrix from .npy file
template <typename T>
void load_npy(const std::string& path, blaze::DynamicMatrix<T, blaze::rowMajor>& x) {
    std::cout << "reading file " << path << "..." << std::endl;
    auto npy_array = cnpy::npy_load(path);
    T* loaded_data = npy_array.data<T>();

    size_t nrows = npy_array.shape[0];
    size_t ncols = npy_array.shape[1];
    
    x.resize(nrows, ncols);
    
    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            x(i, j) = loaded_data[j+ncols*i];
        }
    }
    
    std::cout << "done!" << std::endl;;
    std::cout << "data points N = " << x.rows() << std::endl;
    std::cout << "features    D = " << x.columns() << std::endl;;
}

// write vector of size_t as a npy file
void save_npy(const std::string& path, const std::vector<size_t>& x) {
    cnpy::npy_save(path, &x[0], {x.size()}, "w");
}

// write vector of size_t as a npy file
template <typename T>
void save_npy(const std::string& path, const blaze::DynamicMatrix<T>& x) {
    
    std::vector<T> data;
    for (size_t i = 0; i < x.rows(); i++) {
        for (size_t j = 0; j < x.columns(); j++) {
            data.emplace_back(x(i, j));
        }
    }
    
    cnpy::npy_save(path, &data[0], {data.size()}, "w");
}

// reads a blaze matrix from whitespace separated text file
template <typename T>
void load_txt(const std::string& path, blaze::DynamicMatrix<T, blaze::rowMajor>& x) {
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

// write vector of size_t as a text file
void save_txt(const std::string& path, const std::vector<size_t>& x) {
    std::ofstream ofs(path);
    for (const auto& index : x) {
        ofs << index << "\n";
    }
}

// writes blaze matrix as a text file
template <typename T>
void save_txt(const std::string& path, const blaze::DynamicMatrix<T>& x) {
    std::ofstream ofs(path);

    for (size_t i = 0; i < x.rows(); i++) {
        for (size_t j = 0; j < x.columns(); j++) {
            ofs << x(i, j) << "\t";
        }
        ofs << "\n";
    }
}


#endif
