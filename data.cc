#include "../include/data.hpp"

void data::set_feature_vector(std::vector<double> *vect){
    feature_vector = vect;
}

void data::append_to_feature_vector(double vect){
    feature_vector->push_back(vect);
}

void data::set_original_label(const std::string &label){
    original_label = label;
}

//Set Vectors, distance, and labels
void data::set_label(uint8_t lab){
    label = lab;
}

void data::set_distance(double val){
    distance = val;
}

void data::set_class_vector(int counts){
    class_vector = new std::vector<int>();
    for(int i = 0; i < counts; i++){
        if(i == label)
            class_vector->push_back(1);
        else
            class_vector->push_back(0);
    }
}

void data::set_enum_label(int lab){
    enum_label = lab;
}

std::string data::get_original_label(){
    return original_label;
}

//Get Distance, and Labels
double data::get_distance(){
    return distance;
}

double data::get_label(){
    return label;
}

int data::get_enum_label(){
    return enum_label;
}

//Getters for vectors
std::vector<double> * data::get_feature_vector(){
    return feature_vector;
}

std::vector<int> * data::get_class_vector(){
    return class_vector;
}