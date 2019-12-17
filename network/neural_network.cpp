//
#include <iostream>
#include <math.h>

const int image_height = 2;
const int image_width = 1;
const int entry_layer_neurons = image_width * image_height;
const int hidden_layer_neurons = 2;
const int layers_number = 3;


const float learning_const = 0.1;
const int classes = 2; // exit layer neurons


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
//    return x;
}

float transfer_derivative(float x) {
    float output =  x * (1.0 - x);
//    float output = 1.0;
    return output;
}

float generate_random_weight() {
    return ((rand() % 200) / 100.0) - 1.0 ;
//    return 1.0;
}

float generate_random_bias(){
    return generate_random_weight();
//    return 0;
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
        new_network.layers[1].neurons[i].bias = generate_random_bias();

    }
    std::cout << "network initializing!7" << std::endl;

    for (int i = 0; i < classes; i++) {
        new_network.layers[2].neurons[i].activation = 0;
        for (int j = 0; j < hidden_layer_neurons; j++) {
            new_network.layers[2].neurons[i].weights[j] = generate_random_weight();
//            new_network.layers[2].neurons[i].weights[j] = 0.0;

        }
        new_network.layers[2].neurons[i].bias = generate_random_bias();
    }
    std::cout << "network initializing!8" << std::endl;

    return new_network;
}


void display_network(network network1) {
    for (int layer_id = 0; layer_id < layers_number; layer_id++) {
        std::cout << "layer displaying: " << layer_id << std::endl;
        if(layer_id ==0){
            for(int i=0; i<entry_layer_neurons; i++){
                std::cout<<network1.layers[0].neurons[i].activation;
            }
        }
        if (layer_id == 1) {
            for (int j = 0; j < hidden_layer_neurons; j++) {
                for (int k = 0; k < entry_layer_neurons; k++) {
                    std::cout << network1.layers[1].neurons[j].weights[k] << layer_id << std::endl;
                }
            }
        }if(layer_id ==2){
            for(int k=0; k<classes; k++){
                for (int x = 0; x<hidden_layer_neurons; x++){
                    std::cout<<network1.layers[2].neurons[k].weights[x] << layer_id << std::endl;
                }
            }
        }

    }

}


void calculate_neuron_activation(network &our_network, int layer_number, neuron &neuron1) {
    int previous_layer_neurons;
    if (layer_number == 1) {
        previous_layer_neurons = entry_layer_neurons;
    }
    else if (layer_number == 2) {
        previous_layer_neurons = hidden_layer_neurons;
    } else{
        return;
    }

    float activation = 0;
    for (int i = 0; i < previous_layer_neurons; i++) {
        activation += neuron1.weights[i] * our_network.layers[layer_number - 1].neurons[i].activation;
    }
    activation += neuron1.bias;
//    activation = activation / (previous_layer_neurons + 1);
    activation = sigmoid(activation);
    neuron1.activation = activation;
}

void calculate_output(network &our_network) {
    for (int i = 0; i < classes; i++) {
        calculate_neuron_activation(our_network, 2, our_network.layers[2].neurons[i]);
    }
}

int choose_result(network our_network) {
    float max_activation = 0.0;
    int class_number = 0;
    for (int i = 0; i < classes; i++) {
        if (our_network.layers[2].neurons[i].activation > max_activation) {
            max_activation = our_network.layers[2].neurons[i].activation;
            class_number = i;
        };
    }
    std::cout<<"zgodność "<<max_activation<<std::endl;
    return class_number;
}

void backpropagate(network &our_network, int expected_class) {

    // calculate Errors
    float exit_layer_errors[classes];
    float hidden_layer_errors[hidden_layer_neurons];

    for(int i=0; i<classes; i++){
        exit_layer_errors[i] = 0.0;
    }

    for(int i=0; i<hidden_layer_neurons; i++){
        hidden_layer_errors[i] = 0.0;
    }


    float expected;
    float output;
    for(int i=0; i<classes; i++){
        if(expected_class == i){
            expected = 1;
        } else{
            expected = 0;
        }
        output = our_network.layers[2].neurons[i].activation;
//        float xd = (expected - output) * transfer_derivative(output);
        exit_layer_errors[i] += (expected - output) * transfer_derivative(output);
    }
    float error;
    for(int i=0; i<hidden_layer_neurons; i++){
        error = 0.0;
        output = our_network.layers[1].neurons[i].activation;
        for(int j=0; j<classes; j++){
            neuron next_layer_neuron = our_network.layers[2].neurons[j];
            error += (next_layer_neuron.weights[i] * exit_layer_errors[j]) * transfer_derivative(output);  // "/classes?
        }
        hidden_layer_errors[i] = error;
    }
    //Errors calculated

    //Update weights
    for (int i=0; i<classes; i++){
//        neuron& updated_neuron = our_network.layers[2].neurons[i];
        neuron updated_neuron = our_network.layers[2].neurons[i];
        float change = 0;
        float old = 0;

        for(int j=0; j< hidden_layer_neurons; j++){
            change = updated_neuron.weights[j] + exit_layer_errors[i] * updated_neuron.activation * learning_const;
            old = updated_neuron.weights[j];
            updated_neuron.weights[j] = change ;
        }
        updated_neuron.bias = updated_neuron.bias + exit_layer_errors[i] * updated_neuron.activation * learning_const;
        our_network.layers[2].neurons[i] = updated_neuron;
    }

    for (int i=0; i<hidden_layer_neurons; i++){
//        neuron& updated_neuron = our_network.layers[1].neurons[i];
        neuron updated_neuron = our_network.layers[1].neurons[i];
        float change = 0;
        for(int j=0; j<entry_layer_neurons; j++){
            change = updated_neuron.weights[j] + hidden_layer_errors[i] * updated_neuron.activation * learning_const;
            updated_neuron.weights[j]=change;
        }
        updated_neuron.bias = updated_neuron.bias + hidden_layer_errors[i] * updated_neuron.activation * learning_const;
        our_network.layers[1].neurons[i] = updated_neuron;
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


void learn(float pixels[entry_layer_neurons], network &our_network, int class_number, bool write){
    load_image(pixels, our_network);
    for (int i=0; i<hidden_layer_neurons; i++){
        calculate_neuron_activation(our_network, 1, our_network.layers[1].neurons[i]);
    }
//    for (int i=0; i<hidden_layer_neurons; i++){
////        std::cout<<our_network.layers[1].neurons[i].activation<<std::endl;
//    }

    calculate_output(our_network);
    int result = choose_result(our_network);
    if(write) {
        std::cout << "wanted " << class_number << "got " << result << std::endl;
    }
    backpropagate(our_network, class_number);
}

void check_result(float pixels[entry_layer_neurons], network &our_network, int class_number, bool write){
    load_image(pixels, our_network);
    for (int i=0; i<hidden_layer_neurons; i++){
        calculate_neuron_activation(our_network, 1, our_network.layers[1].neurons[i]);
    }
//    for (int i=0; i<hidden_layer_neurons; i++){
////        std::cout<<our_network.layers[1].neurons[i].activation<<std::endl;
//    }

    calculate_output(our_network);
    int result = choose_result(our_network);
    if(write) {
        std::cout << "wanted " << class_number << "got " << result << std::endl;
    }
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

    for(int i=0; i<10000000; i++){
////        learn(random_arr, our_network, 1, false);
        learn(random_arr3, our_network, 1, true);
        learn(random_arr2, our_network, 2, true);
//        learn(false1, our_network, 0, true);
//        learn(false2, our_network, 0, true);
//        learn(true1, our_network, 1, true);
//        learn(true2, our_network, 1, true);
    }



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
