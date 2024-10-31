#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "common.hpp"
#include "data_handler.hpp"
#include <cmath>
#include <iostream>
#include <fstream>

class network : public common_data{
    private:
        int input_size;
    public:
        std::vector<layer *> layers;
        double learning_rate;
        double test_performance;
        network(std::vector<int> spec, int input_size, int num_classes, double learning_rate);
        ~network();
        std::vector<double> fprop(data * d);
        double activate(std::vector<double> weights, std::vector<double> input);
        double transfer(double activation);
        double transfer_derivative(double output);
        void bprop(data * d);
        void update_weights(data *d);
        int predict(data *d);
        void train(int number_of_epochs);
        double test();
        void validate();
        void save_model(const std::string &filename);
        void load_model(const std::string &filename);
        void output_predictions(const std::string &filename, data_handler *dh, data *user_input_data = nullptr);
        std::vector<double> get_user_input();
};
#endif