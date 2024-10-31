#ifndef __COMMON_HPP
#define __COMMON_HPP

#include "data.hpp"
#include <vector>

class common_data{
    protected:
        std::vector<data *> * common_training_data;
        std::vector<data *> * common_testing_data;
        std::vector<data *> * common_validation_data;
    public:
        void set_common_training_data(std::vector<data *> *vect);
        void set_common_testing_data(std::vector<data *> *vect);
        void set_common_validation_data(std::vector<data *> *vect);
};
#endif