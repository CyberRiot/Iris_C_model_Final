#include "layer.hpp"

layer::layer(int previous_layer_size, int current_layer_size){
    for(int i = 0; i < current_layer_size; i++){
        neurons.push_back(new neuron(previous_layer_size, current_layer_size));
    }
    this->current_layer_size = current_layer_size;
}