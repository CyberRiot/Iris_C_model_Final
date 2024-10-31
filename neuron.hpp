#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <cmath>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

class neuron{
    public:
        double output;
        double delta;
        std::vector<double> weights;

        double generate_random_number(double min, double max);
        neuron(int previous_layer_size, int current_layer_size);
        void initialize_weights(int previous_layer_size);

        void save_weights(std::ofstream &out);
        void load_weights(std::ifstream &in);
};

#endif