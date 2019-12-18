//
#include <iostream>
#include <math.h>
#include <algorithm>


const int image_height = 2;
const int image_width = 1;
const int entry_layer_neurons = image_width * image_height;
const int hidden_layers_number = 1;
const int output_layer_neurons = 2; // exit layer neurons

const int layer_neurons_number[] = {entry_layer_neurons,16, output_layer_neurons};
const int layers_number = hidden_layers_number +2;

const float learning_const = 0.1;
const int max_neurons_in_layer = layer_neurons_number[0];



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
    return 1 / (1 + exp(-x));
}

float transfer_derivative(float x) {
    return  x * (1.0 - x);
}

float generate_random_weight() {
    return ((rand() % 200) / 100.0) - 1.0 ;
}

float generate_random_bias(){
    return generate_random_weight();
}

network initialize() {
    std::cout << "network initializing!" << std::endl;
    network new_network;
    std::cout << "layers created" << std::endl;

    //Initialize layers

    for (int i=0; i< layers_number; i++){
        new_network.layers[i].neurons = new neuron[layer_neurons_number[i]];
        for (int j = 0; j<layer_neurons_number[i]; j++) {
            new_network.layers[i].neurons[j].activation = 0;
            if (i > 0) {
                new_network.layers[i].neurons[j].weights = new float[layer_neurons_number[i-1]];
                for (int k = 0; k < layer_neurons_number[i - 1]; k++) {
                    new_network.layers[i].neurons[j].weights[k] = generate_random_weight();
                }
                new_network.layers[i].neurons[j].bias = generate_random_bias();
            }
        }
    }
    std::cout << "neurons and weight initialized" << std::endl;


    //DEBUG
    for (int i = 0; i < entry_layer_neurons; i++) {
        std::cout << new_network.layers[0].neurons[i].activation << std::endl;
    }
    // !DEBUG

    std::cout << "whole network initialized" << std::endl;


    return new_network;
}


void display_network(network network1) {
    for (int current_layer = 0; current_layer<layers_number; current_layer ++){
        for (int j=0; j<layer_neurons_number[current_layer]; j++){
            std::cout<<network1.layers[current_layer].neurons[j].activation<< std::endl;
            if(current_layer>0){
                std::cout<<"weights"<<std::endl;
                for(int previous_layer_neuron =0; previous_layer_neuron< layer_neurons_number[current_layer-1]; previous_layer_neuron++){
                    std::cout<<network1.layers[current_layer].neurons[j].weights[previous_layer_neuron] << std::endl;
                }
            }
        }
    }
}


void calculate_neuron_activation(network &our_network, int layer_number, neuron &neuron1) {

    if (layer_number == 0) {
        return;
    }

    float activation = 0;
    for (int i = 0; i < layer_neurons_number[layer_number-1]; i++) {
        activation += neuron1.weights[i] * our_network.layers[layer_number - 1].neurons[i].activation;
    }
    activation += neuron1.bias;
    activation = sigmoid(activation);
    neuron1.activation = activation;
}

//void calculate_output(network &our_network) {
//    for (int i = 0; i < output_layer_neurons; i++) {
//        calculate_neuron_activation(our_network, layers_number-1, our_network.layers[2].neurons[i]);
//    }
//}

int choose_result(network our_network) {
    float max_activation = 0.0;
    int class_number = 0;
    for (int i = 0; i < output_layer_neurons; i++) {
        if (our_network.layers[layers_number-1].neurons[i].activation > max_activation) {
            max_activation = our_network.layers[layers_number-1].neurons[i].activation;
            class_number = i;
        };
    }
    std::cout<<"zgodność "<<max_activation<<std::endl;
    return class_number;
}

void backpropagate(network &our_network, int expected_class) {

    // calculate Errors
    float errors[layers_number-1][max_neurons_in_layer]; // = new float[

    for(int i=layers_number-1;i>0;i--){
        for(int j=0;j<layer_neurons_number[i];j++)
            errors[i][j]=0.0;
    }

    float expected;
    // find class number
    for(int i=0; i<layer_neurons_number[layers_number-1]; i++) {
        if (expected_class == i) {
            expected = 1;
        } else {
            expected = 0;
        }
    }

    float output;
    for(int i=0; i<output_layer_neurons; i++){
        if(expected_class == i){
            expected = 1;
        } else{
            expected = 0;
        }
        output = our_network.layers[layers_number-1].neurons[i].activation;
        exit_layer_errors[i] += (expected - output) * transfer_derivative(output);
    }

    float error;
    for(int i=0; i<layer_neurons_number; i++){
        error = 0.0;
        output = our_network.layers[1].neurons[i].activation;
        for(int j=0; j<output_layer_neurons; j++){
            neuron next_layer_neuron = our_network.layers[2].neurons[j];
            error += (next_layer_neuron.weights[i] * exit_layer_errors[j]) * transfer_derivative(output);  // "/classes?
        }
        hidden_layer_errors[i] = error;
    }
    //Errors calculated

    //Update weights
    for (int i=0; i<layer_neurons_number; i++){
//        neuron& updated_neuron = our_network.layers[1].neurons[i];
        neuron updated_neuron = our_network.layers[1].neurons[i];
        float change = 0;
        for(int j=0; j<entry_layer_neurons; j++){
            change = updated_neuron.weights[j] + hidden_layer_errors[i] * our_network.layers[0].neurons[j].activation * learning_const;
            updated_neuron.weights[j]=change;
        }
        updated_neuron.bias = updated_neuron.bias + hidden_layer_errors[i] * updated_neuron.activation * learning_const;
        our_network.layers[1].neurons[i] = updated_neuron;
    }

    for (int i=0; i<output_layer_neurons; i++){
//        neuron& updated_neuron = our_network.layers[2].neurons[i];
        neuron updated_neuron = our_network.layers[2].neurons[i];
        float change = 0;
        float old = 0;

        for(int j=0; j< layer_neurons_number; j++){
            change = updated_neuron.weights[j] + exit_layer_errors[i] * our_network.layers[1].neurons[j].activation * learning_const;
            old = updated_neuron.weights[j];
            updated_neuron.weights[j] = change ;
        }
        updated_neuron.bias = updated_neuron.bias + exit_layer_errors[i] * updated_neuron.activation * learning_const;
        our_network.layers[2].neurons[i] = updated_neuron;
    }
}

void load_image(float pixels[entry_layer_neurons], network &our_network){
    //inicializuje ok
    for(int i=0; i<entry_layer_neurons; i++){
        our_network.layers[0].neurons[i].activation=pixels[i];
    }

//    for(int i=0; i<entry_layer_neurons; i++){
//        std::cout<<our_network.layers[0].neurons[i].activation<<std::endl;
//    }
    return;
}




void check_result(float pixels[entry_layer_neurons], network &our_network, int class_number, bool write){
    load_image(pixels, our_network);
    for(int j=1;j<layers_number;j++){
        for (int i=0; i<layer_neurons_number[j]; i++){
            calculate_neuron_activation(our_network, j, our_network.layers[j].neurons[i]);
        }
    }
//    for (int i=0; i<hidden_layer_neurons; i++){
////        std::cout<<our_network.layers[1].neurons[i].activation<<std::endl;
//    }

//    calculate_output(our_network);
    if(write) {
        int result = choose_result(our_network);
        std::cout << "wanted " << class_number << "got " << result << std::endl;
    }
}

void learn(float pixels[entry_layer_neurons], network &our_network, int class_number, bool write){
//    load_image(pixels, our_network);
    check_result(pixels, our_network, class_number, false);
    int result = choose_result(our_network);
    if(write) {
        std::cout << "wanted " << class_number << "got " << result << std::endl;
    }
    backpropagate(our_network, class_number);
}

int main() {
    srand( time( 0 ) );
    float random_arr[entry_layer_neurons];
    float random_arr2[entry_layer_neurons];
    float random_arr3[entry_layer_neurons];

    for (int i=0; i<entry_layer_neurons; i++){
//        random_arr[i] = ((rand()%50)/100.0 + 0.5);
        random_arr2[i] = 2;
        random_arr3[i] = -1;

    }
//    float false1[] = {0,0};
//    float false2[] = {1,1};
//    float true1[] = {0,1};
//    float true2[] = {1,0};
    network our_network = initialize();

//    our_network.layers[1].neurons[0].weights[0] = - 1;
//    learn(random_arr, our_network, 0, true);
//    learn(random_arr3, our_network, 1, true);
//    learn(random_arr2, our_network, 2, true);

    for(int i=0; i<1000000; i++){
////        learn(random_arr, our_network, 1, false);
        learn(random_arr3, our_network, 1, true);
        learn(random_arr2, our_network, 0, true);
//        learn(false1, our_network, 0, true);
//        learn(false2, our_network, 0, true);
//        learn(true1, our_network, 1, true);
//        learn(true2, our_network, 1, true);
    }
//    learn(random_arr3, our_network, 1, true);
//    learn(random_arr2, our_network, 0, true);


//    check_result(false1, our_network, 0, true);
//    check_result(false2, our_network, 0, true);
//    check_result(true1, our_network, 1, true);
//    check_result(true2, our_network, 1, true);
//    learn(random_arr, our_network, 1, true);
//    learn(random_arr3, our_network, 3, true);
//    learn(random_arr2, our_network, 2, true);
    display_network(our_network);

    return 0;
}
