#include <network.hpp>

network::network(std::vector<int> spec, int input_size, int num_classes, double learning_rate){
    this->input_size = input_size;
    for(int i = 0; i < spec.size(); i++){
        if(i == 0)
            layers.push_back(new layer(input_size, spec.at(i)));
        else
            layers.push_back(new layer(layers.at(i-1)->neurons.size(), spec.at(i)));
    }
    layers.push_back(new layer(layers.at(layers.size()-1)->neurons.size(), num_classes));
    this->learning_rate = learning_rate;
}

network::~network(){}

std::vector<double> network::fprop(data * d){
    std::vector<double> inputs = *d->get_feature_vector();
    for(int i = 0; i < layers.size(); i++){
        layer *l = layers.at(i);
        std::vector<double> new_inputs;
        for(neuron *n : l->neurons){
            double activation = this->activate(n->weights, inputs);
            n->output = this->transfer(activation);
            new_inputs.push_back(n->output);
        }
        inputs = new_inputs;
    }
    return inputs;
}

double network::activate(std::vector<double> weights, std::vector<double> input){
    double activation = weights.back();
    for(int i = 0; i < weights.size() - 1; i++){
        activation += weights[i] * input[i];
    }
    return activation;
}

double network::transfer(double activation){
    return 1.0 / (1.0 + exp(-activation));
}

double network::transfer_derivative(double output){
    return output * (1 - output);
}

void network::bprop(data *d){
    for(int i = layers.size() - 1; i >= 0; i--){
        layer *l = layers.at(i);
        std::vector<double> errors;
        if(i != layers.size() - 1){
            for(int j = 0; j < l->neurons.size(); j++){
                double error = 0.0;
                for(neuron *n : layers.at(i + 1)->neurons){
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        }
        else{
            for(int j = 0; j < l->neurons.size(); j++){
                neuron *n = l->neurons.at(j);
                errors.push_back((double)d->get_class_vector()->at(j) - n->output);
            }
        }
        for(int j = 0; j < l->neurons.size(); j++){
            neuron *n = l->neurons.at(j);
            n->delta = errors.at(j) * this->transfer_derivative(n->output);
        }
    }
}

void network::update_weights(data *d){
    std::vector<double> inputs = *d->get_feature_vector();
    for(int i = 0; i < layers.size(); i++){
        if(i != 0){
            for(neuron *n : layers.at(i - 1)->neurons){
                inputs.push_back(n->output);
            }
        }
        for(neuron *n : layers.at(i)->neurons){
            for(int j = 0; j < inputs.size(); j++){
                n->weights.at(j) += this->learning_rate * n->delta * inputs.at(j);
            }
            n->weights.back() += this->learning_rate * n->delta;
        }
        inputs.clear();
    }
}

int network::predict(data *d){
    std::vector<double> outputs = fprop(d);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void network::train(int number_of_epochs){
    for(int i = 0; i < number_of_epochs; i++){
        double sum_error = 0.0;
        for(data *d : *this->common_training_data){
            std::vector<double> outputs = fprop(d);
            std::vector<int> expected = *d->get_class_vector();
            double temp_error_sum = 0.0;
            for(int j = 0; j < outputs.size(); j++){
                temp_error_sum += pow((double) expected.at(j) - outputs.at(j), 2);
            }
            sum_error += temp_error_sum;
            bprop(d);
            update_weights(d);
        }
        printf("\rIteration %d \t Error %.4f", i+1, sum_error);
        fflush(stdout);

        //Print Progress Bar
        int bar_width = 50;
        printf("\t\t\tProgress: [");
        int pos = bar_width * (i + 1) / number_of_epochs;
        for(int j = 0; j < bar_width; j++){
            if(j < pos) std::cout << "=";
            else if(j == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int((i + 1) * 100.0 / number_of_epochs) << " %";
        fflush(stdout);
    }
    std::cout << std::endl;
}

double network::test(){
    double number_correct = 0.0;
    double count = 0.0;
    for(data *d : *this->common_testing_data){
        count++;
        int index = predict(d);
        if(d->get_class_vector()->at(index) == 1) number_correct++;
    }
    test_performance = (number_correct / count);
    return test_performance;
}

void network::validate(){
    double number_correct = 0.0;
    double count = 0.0;
    for(data *d : *this->common_validation_data){
        count++;
        int index = predict(d);
        if(d->get_class_vector()->at(index) == 1) number_correct++;
    }
    printf("Validation Performance: %.2f\t", number_correct / count);
}

void network::save_model(const std::string &filename){
    std::ofstream out(filename);
    if(!out.is_open()){
        std::cerr << "Error opening file to save model: " << filename << std::endl;
        return;
    }

    //Save Network Architecture
    out << layers.size() << std::endl;
    for(layer *l : layers){
        out << l->current_layer_size << " ";
    }
    out << std::endl;

    //Save Weights for each neuron in each layer
    for(layer *l : layers){
        for(neuron *n : l->neurons){
            n->save_weights(out);
        }
    }
    out.close();
}

void network::load_model(const std::string &filename){
    std::ifstream in(filename);
    if(!in.is_open()){
        std::cerr << "Error opening file for loading model: " << filename << std::endl;
        return;
    }

    // Load Network Architecture
    int num_layers;
    in >> num_layers;

    std::vector<int> layer_sizes(num_layers);
    for(int i = 0; i < num_layers; i++){
        in >> layer_sizes[i];
    }

    // Initialize layers based on loaded architecture
    layers.clear();
    for(int i = 0; i < num_layers; i++){
        int previous_layer_size = (i == 0) ? input_size : layer_sizes[i - 1];
        layers.push_back(new layer(previous_layer_size, layer_sizes[i]));
    }

    // Skip the newline character after the last layer size
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Load Weights for each Neuron in each Layer
    for(layer *l : layers){
        for(neuron *n : l->neurons){
            n->load_weights(in);
        }
    }

    in.close();
}


std::vector<double>* get_user_input(){
    auto input = new std::vector<double>;
    double value;

    std::cout << "Enter the 4 values (seperated by a space): " << std::endl;
    std::cout << "Example: [0]Sepal Length, [1]Sepal Width, [2]Pedal Length, [3]Pedal Width." << std::endl;

    for(int i = 0; i < 4; i++){
        std::cin >> value;
        input->push_back(value);
    }
    return input;
}

void network::output_predictions(const std::string &filename, data_handler *dh, data *user_input_data) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error opening file for outputting predictions: " << filename << std::endl;
        return;
    }

    // Write header
    out << "SepalLength,SepalWidth,PetalLength,PetalWidth,Label\n";

    // Write data and predictions for each sample in the dataset
    for (data *d : *dh->get_full_data()) {
        std::vector<double> features = *d->get_feature_vector();

        // Write features
        for (double feature : features) {
            out << feature << ",";
        }

        // Write original label
        out << d->get_original_label() << "\n";
    }

    // Write the user input if provided
    if (user_input_data != nullptr) {
        std::vector<double> features = *user_input_data->get_feature_vector();

        // Write features
        for (double feature : features) {
            out << feature << ",";
        }

        // Write label (UserInput)
        out << "UserInput\n";
    }

    out.close();
}

//The Iris Dataset is primarily used to practice data analysis, machine learning and data visualization
//The iris dataset that has been ingested is done in this order :
//[0]Sepal Length, [1]Sepal Width, [2]Pedal Length, [3]Pedal Width, [4]Label

int main(){
    data_handler *dh = new data_handler;
    dh->read_csv("../iris.data", ",");
    dh->split_data();
    std::vector<int> hidden_layers = {150};
    auto lambda = [&]() {
    network *net = new network(hidden_layers, dh->get_training_data()->at(0)->get_feature_vector()->size(), dh->get_class_counts(), 0.25);

    // Set data
    net->set_common_training_data(dh->get_training_data());
    net->set_common_testing_data(dh->get_testing_data());
    net->set_common_validation_data(dh->get_validation_data());

    // Debugging output
    std::cout << "Data set. Before model load.\n";

    // Check if the saved model exists
    std::string model_filename = "saved_model.txt";
    std::ifstream infile(model_filename);
    if (infile.good()) {
        net->load_model(model_filename);
        printf("Model loaded from %s\n", model_filename.c_str());
    } else {
        net->train(5000);
        net->save_model(model_filename);
        printf("Model saved to %s\n", model_filename.c_str());
    }
    net->validate();

    // Debugging output
    std::cout << "Model validated.\n";

    printf("\t\tTest Performance: %.3f\n", net->test());

    // Get User Input Here
    std::vector<double>* new_data = get_user_input();

    // Create a Data Object for Prediction
    data *new_data_obj = new data();
    new_data_obj->set_feature_vector(new_data);

    int predicted_class = net->predict(new_data_obj);
    std::cout << "Predicted Class: " << predicted_class << std::endl;
    if(predicted_class == 0){
        printf("Iris Setosa.\n");
    }
    else if(predicted_class == 1){
        printf("Iris Versicolor.\n");
    }
    else if(predicted_class == 2){
        printf("Iris Virginica.\n");
    }
    else{
        printf("Invalid Assessment.\n");
    }

    //Output predictions to a CSV file
    std::string output_filename = "predictions.csv";
    net->output_predictions(output_filename, dh, new_data_obj);
    std::cout << "Predictions saved to: " << output_filename << std::endl;

    // Clean Up
    delete new_data_obj;
    delete net;
    };
    lambda();

    delete dh;
    return 0;
}