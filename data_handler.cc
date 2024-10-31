#include "../include/data_handler.hpp"

//Constructor and Deconstrutor
data_handler::data_handler(){
    data_array = new std::vector<data *>;
    testing_data = new std::vector<data *>;
    training_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}

data_handler::~data_handler(){
    free(data_array);
    free(testing_data);
    free(training_data);
    free(validation_data);
}

//Read Iris Data set
void data_handler::read_csv(std::string path, std::string delimiter){
    class_counts = 0;
    std::ifstream data_file;

    data_file.open(path.c_str());
    std::string line;

    while(std::getline(data_file, line)){
        if(line.length() == 0) continue;
        data *d = new data();
        d->set_feature_vector(new std::vector<double>());

        size_t position = 0;
        std::string token;
        while((position = line.find(delimiter)) != std::string::npos){
            token = line.substr(0, position);
            d->append_to_feature_vector(std::stod(token));
            line.erase(0, position + delimiter.length());
        }
        d->set_original_label(line);
        if(class_from_string.find(line) != class_from_string.end()){
            d->set_label(class_from_string[line]);
        }
        else{
            class_from_string[line] = class_counts;
            d->set_label(class_from_string[token]);
            class_counts++;
        }
        data_array->push_back(d);
    }
    for(data * da : *data_array)
        da->set_class_vector(class_counts);;
    feature_vector_size = data_array->at(0)->get_feature_vector()->size();
}

void data_handler::split_data(){
    std::unordered_set<int> used_indexes;
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int testing_size = data_array->size() * TESTING_SET_PERCENT;
    int validation_size = data_array->size() * VALIDATION_SET_PERCENT;

    std::random_shuffle(data_array->begin(), data_array->end());

    //TRAINING DATA
    int count = 0;
    int index = 0;
    while(count < train_size){
        training_data->push_back(data_array->at(index++));
        count++;
    }

    //TESTING DATA
    count = 0;
    while(count < testing_size){
        testing_data->push_back(data_array->at(index++));
        count++;
    }

    //VALIDATION DATA
    count = 0;
    while(count < validation_size){
        validation_data->push_back(data_array->at(index++));
        count++;
    }
    printf("Training Data Size: %lu.\tTesting Data Size: %lu.\tValidation Data Size: %lu.\n", training_data->size(),testing_data->size(),validation_data->size());
}

void data_handler::count_classes(){
    int count = 0;
    for(unsigned i = 0; i < data_array->size(); i++){
        if(class_from_int.find(data_array->at(i)->get_label()) == class_from_int.end()){
            class_from_int[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enum_label(count);
            count++;
        }
        else{
            data_array->at(i)->set_enum_label(class_from_int[data_array->at(i)->get_label()]);
        }
    }
    class_counts = count;
    for(data *da : *data_array)
        da->set_class_vector(class_counts);
    printf("Successfully Extracted %d Unique Classes.\n", class_counts);
}

int data_handler::get_class_counts(){
    return class_counts;
}

std::vector<data *> *data_handler::get_training_data(){
    return training_data;
}

std::vector<data *> *data_handler::get_testing_data(){
    return testing_data;
}

std::vector<data *> *data_handler::get_validation_data(){
    return validation_data;
}

std::vector<data *> *data_handler::get_full_data(){
    std::vector<data *> *full_data = new std::vector<data *>;
    full_data->insert(full_data->end(), training_data->begin(), training_data->end());
    full_data->insert(full_data->end(), testing_data->begin(), testing_data->end());
    full_data->insert(full_data->end(), validation_data->begin(), validation_data->end());
    return full_data;
}