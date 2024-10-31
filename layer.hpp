#ifndef __LAYER_HPP
#define __LAYER_HPP

#include <neuron.hpp>
#include <stdint.h>
#include <vector>

class layer{
    public:
        int current_layer_size;
        std::vector<neuron *> neurons;
        std::vector<double> layer_outputs;
        layer(int previous_layer_size, int current_layer_size);
};

#endif