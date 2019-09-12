/******************************************************************************/
// events_gmm.cxx
//
// Created by Omar Oubari
// Email: omar.oubari@inserm.fr
//
// Information: Applying Florian Hirschberger's vc-GMM on event-based data from neuromorphic cameras
/******************************************************************************/

#include <string>
#include "clustering.hxx"

int main(int argc, char** argv) {
    
    if (argc < 11) {
        throw std::runtime_error(std::to_string(argc).append("received. Expected 11 arguments"));
    } else {
        
        // APPLICATION PARAMETERS
        std::string path_train_features = argv[1]; // path to training features text file
        std::string path_test_features  = argv[2]; // path to training features text file
        int C_p                         = std::atoi(argv[3]); // number of clusters considered for each data point (1 <= C_p <= C)
        int G                           = std::atoi(argv[4]); // search space (nearest neighbours for the C' clusters with 2 <= G <= C)
        bool plus1                      = std::atoi(argv[5]); // include one additional randomly chosen cluster to each of the search spaces
        int N_core                      = std::atoi(argv[6]); // size of subset
        int C                           = std::atoi(argv[7]); // number of cluster centers
        int chain_length                = std::atoi(argv[8]); // chain length for AFK-MC² seeding
        double convergence_threshold    = std::stod(argv[9]); // convergence threshold
        bool save                       = std::atoi(argv[10]); // write cluster centers to a text file
        int nthreads = std::thread::hardware_concurrency(); // number of C++11 threads
        int seed = 123; // seed for random number generator
        
        // READING DATA
        blaze::DynamicMatrix<double, blaze::rowMajor> train_features;
        loadtxt(path_train_features, train_features); // read dataset
        
        // FITTING MODEL TO TRAINING DATASET
        clustering<double> vc_GMM(train_features, N_core, C, chain_length, C_p, G, plus1, nthreads, seed);

        vc_GMM.converge(convergence_threshold); // computes EM-iterations until the relative change in free energy falls below 1e-4
        std::cout << "quantization error: " << vc_GMM.error() << std::endl; // quantization error
        
        // writing the cluster centers to a text file
        if (save) {
            savetxt("cluster_centers.txt", vc_GMM.cluster_centers());
        }
    }
}
