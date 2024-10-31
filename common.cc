#include "common.hpp"

void common_data::set_common_training_data(std::vector<data *> *vect){
    common_training_data = vect;
}

void common_data::set_common_testing_data(std::vector<data *> *vect){
    common_testing_data = vect;
}

void common_data::set_common_validation_data(std::vector<data *> *vect){
    common_validation_data = vect;
}