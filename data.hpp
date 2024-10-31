#ifndef __DATA_HPP
#define __DATA_HPP

#include <vector>
#include "stdint.h"
#include "stdio.h"
#include <iostream>

class data{
    std::vector<double> * feature_vector;
    std::vector<int> * class_vector;
    uint8_t label;
    int enum_label; // A->1, B->2
    double distance;
    std::string original_label;

    public:
        //Set and adjust feature vector
        void set_feature_vector(std::vector<double> *vect);
        void append_to_feature_vector(double vect);

        //Set Original Label
        void set_original_label(const std::string &label);

        //Set Vectors, distance, and labels
        void set_label(uint8_t lab);
        void set_distance(double val);
        void set_class_vector(int counts);
        void set_enum_label(int lab);

        //Get Original Label
        std::string get_original_label();

        //Get Distance, and Labels
        double get_distance();
        double get_label();
        int get_enum_label();

        //Getters for vectors
        std::vector<double> * get_feature_vector();
        std::vector<int> * get_class_vector();        
};
#endif