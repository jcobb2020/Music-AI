//
#include <iostream>
#include <math.h>

const int image_height = 30;
const int image_width = 30;
const int entry_layer_neurons = image_width * image_height;
const int hidden_layer_neurons = 16;
const int layers_number = 3;
const float learning_const = 0.01;

const int classes = 4; // exit layer neurons


struct neuron {
    float activation;
    float bias;
    float *weights;
};

struct layer {
    int id;
    neuron *neurons;
};

struct network {
    layer layers[layers_number];
};

float sigmoid(float x) {
    return x / (1 + exp(-x));
}

float transfer_derivative(float x) {
    return x * (1 - x);
}

float generate_random_weight() {
    return (rand() % 100) / 100.0;
}

network initialize() {
    std::cout << "network initializing!" << std::endl;
    network new_network;

    layer layers[layers_number];
    std::cout << "network initializing!2" << std::endl;

    new_network.layers[0].neurons = new neuron[entry_layer_neurons];
    std::cout << "network initializing!3" << std::endl;

    new_network.layers[1].neurons = new neuron[hidden_layer_neurons];
    std::cout << "network initializing!4" << std::endl;

    new_network.layers[2].neurons = new neuron[classes];
    std::cout << "network initializing!5" << std::endl;

//    for(int i=0; i<entry_layer_neurons; i++){
//        new_network.layers[0].neurons[i].weights;
//    }

//    new_network.layers[0] = layers[0];
//    new_network.layers[1] = layers[1];
//    new_network.layers[2] = layers[2];
    for (int i = 0; i < hidden_layer_neurons; i++) {
        new_network.layers[1].neurons[i].weights = new float[entry_layer_neurons];
    }

    for (int i = 0; i < classes; i++) {
        new_network.layers[2].neurons[i].weights = new float[hidden_layer_neurons];
    }

    for (int i = 0; i < entry_layer_neurons; i++) {
        new_network.layers[0].neurons[i].activation = i;
//        layers[0].neurons[i].activation = 0;
//        for(int j=0; j<)
//        layers[0].neurons[i].weights = generate_random_weight();
    }

    //DEBUG
    for (int i = 0; i < entry_layer_neurons; i++) {
        std::cout << new_network.layers[0].neurons[i].activation << std::endl;
    }
    // !DEBUG

    std::cout << "network initializing!6" << std::endl;

    for (int i = 0; i < hidden_layer_neurons; i++) {
        new_network.layers[1].neurons[i].activation = 0;
        for (int j = 0; j < entry_layer_neurons; j++) {
            new_network.layers[1].neurons[i].weights[j] = generate_random_weight();
        }
    }
    std::cout << "network initializing!7" << std::endl;

    for (int i = 0; i < classes; i++) {
        new_network.layers[2].neurons[i].activation = 0;
        for (int j = 0; j < hidden_layer_neurons; j++) {
            new_network.layers[2].neurons[i].weights[j] = generate_random_weight();
        }
    }
    std::cout << "network initializing!8" << std::endl;

    return new_network;
}


void display_network(network network1) {
    for (int layer_id = 0; layer_id < layers_number; layer_id++) {
        std::cout << "layer displaying: " << layer_id << std::endl;
        if (layer_id == 1) {
            for (int j = 0; j < hidden_layer_neurons; j++) {
                for (int k = 0; k < entry_layer_neurons; k++) {
                    std::cout << network1.layers[1].neurons[j].weights[k] << layer_id << std::endl;
                }
            }
        }
    }

}


void calculate_neuron_activation(network our_network, int layer_number, neuron &neuron1) {
    int previous_layer_neurons;
    if (layer_number == 1) {
        previous_layer_neurons = entry_layer_neurons;
    }
    if (layer_number == 2) {
        previous_layer_neurons = hidden_layer_neurons;
    } else { return; }
    float activation = 0;
    for (int i = 0; i < previous_layer_neurons; i++) {
        activation += neuron1.weights[i] * our_network.layers[layer_number - 1].neurons[i].activation;
    }
    activation += neuron1.bias;
    activation = activation / (previous_layer_neurons + 1);
    activation = sigmoid(activation);
    neuron1.activation = activation;
}

void calculate_output(network our_network) {
    for (int i = 0; i < classes; i++) {
        calculate_neuron_activation(our_network, 2, our_network.layers[2].neurons[i]);
    }
}

int choose_result(network our_network) {
    float max = 0.0;
    int class_number = 0;
    for (int i = 0; i < classes; i++) {
        if (our_network.layers[2].neurons[i].activation > max) {
            max = our_network.layers[2].neurons[i].activation;
            class_number = i;
        };
    }
    return class_number;
}

void backpropagate(network &our_network, int expected_class) {

    // calculate Errors
    float exit_layer_errors[classes];
    float hidden_layer_errors[hidden_layer_neurons];

    int expected;
    float output;
    for(int i=0; i<classes; i++){
        if(expected_class == i){
            expected = 1;
        } else{
            expected = 0;
        }
        output = our_network.layers[2].neurons[i].activation;
        exit_layer_errors[i] = (expected - output) * transfer_derivative(output);
    }
    float error;
    for(int i=0; i<hidden_layer_neurons; i++){
        error = 0.0;
        output = our_network.layers[1].neurons[i].activation;
        for(int j=0; j<entry_layer_neurons; j++){
            neuron next_layer_neuron = our_network.layers[2].neurons[j];
            error += (next_layer_neuron.weights[i] * exit_layer_errors[j]) * transfer_derivative(output);
        }
        hidden_layer_errors[i] = error;
    }
    //Errors calculated

    //Update weights
    for (int i=0; i<classes; i++){
        neuron& updated_neuron = our_network.layers[2].neurons[i];
        for(int j=0; j< hidden_layer_neurons; j++){
            updated_neuron.weights[j] = updated_neuron.weights[j] + exit_layer_errors[i] * updated_neuron.activation * learning_const;
        }
    }

    for (int i=0; i<hidden_layer_neurons; i++){
        neuron& updated_neuron = our_network.layers[1].neurons[i];
        for(int j=0; j<entry_layer_neurons; j++){
            updated_neuron.weights[j] = updated_neuron.weights[j] + hidden_layer_errors[i] * updated_neuron.activation * learning_const;
        }
    }
}


int main() {
    network our_network = initialize();
    display_network(our_network);

    return 0;
}



// Created by jcobb on 12/3/19.
//

