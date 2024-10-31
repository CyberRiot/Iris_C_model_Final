#ifndef __DATA_HANDLER_HPP
#define __DATA_HANDLER_HPP

#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <random>

class data_handler{
    std::vector<data *> *data_array;
    std::vector<data *> *training_data;
    std::vector<data *> *testing_data;
    std::vector<data *> *validation_data;

    //ints to control the class info
    int class_counts;
    int num_classes;
    int feature_vector_size;

    std::map<uint8_t, int> class_from_int;
    std::map<std::string, int> class_from_string;

    //TRAINING, TESTING, VALIDATION PERCENTAGES
    const double TRAIN_SET_PERCENT = 0.75;
    const double TESTING_SET_PERCENT = 0.20;
    const double VALIDATION_SET_PERCENT = 0.05;

    public:
        //Constructor and Deconstrutor
        data_handler();
        ~data_handler();

        //Read Iris Data set
        void read_csv(std::string path, std::string delimiter);
        void split_data();
        void count_classes();

        int get_class_counts();

        std::vector<data *> *get_training_data();
        std::vector<data *> *get_testing_data();
        std::vector<data *> *get_validation_data();
        std::vector<data *> *get_full_data();
};
#endif