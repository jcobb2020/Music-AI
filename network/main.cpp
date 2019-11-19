#include <iostream>

const int image_height = 32;
const int image_width = 32;
const int layer_1_neurons = image_height * image_width;
const int layer_2_neurons = 32;
const int layer_3_neurons = 32;

// depends on output classes number
const int exit_layer_neurons = 10;

const int layers_1_2_links_num = layer_1_neurons * layer_2_neurons;
const int layers_2_3_links_num = layer_2_neurons * layer_3_neurons;
const int layers_3_exit_links_num = layer_3_neurons * exit_layer_neurons;

struct neuron{
    double activation;
};

struct link{
    neuron* neuron1;
    neuron* neuron2;
    double weight;
};

struct network{
    neuron entry_layer[layer_1_neurons];
    neuron second_layer[layer_2_neurons];
    link layers_1_2_links[layers_1_2_links_num];
    neuron third_layer[image_height];
    link layers_2_3_links[layers_2_3_links_num];
    neuron exit_layer[exit_layer_neurons];
    link layer_3_exit_links[layers_3_exit_links_num];

};

double sigmoid(float x){
    return x / (1 + abs(x));
}

void count_layer_activation(neuron layer[], int layer_size, link links[], int layer_x_y_links_num){
    float sum = 0;
    int links_per_neuron = layer_x_y_links_num/layer_size;
    for (int i=0; i<layer_size; i++){
        sum = 0;
        for (int j=0; j<links_per_neuron; j++){
            sum+=links[i*links_per_neuron + j].weight * links[i*links_per_neuron + j].neuron1->activation;
        }
        layer[i].activation = sigmoid(sum);
    }
}

double count_activation(neuron neuron1, link link1){
    return sigmoid(neuron1.activation*link1.weight);
}

void display_network(network network1){
    for(int i=0; i<layer_1_neurons; i++){
        std::cout<<network1.entry_layer[i].activation;
        std::cout<<"\n";
    }
    for(int j=0; j<layer_2_neurons; j++){
        std::cout<<network1.second_layer[j].activation;
        std::cout<<"\n";
    }
    for(int k=0; k<layer_3_neurons; k++){
        std::cout<<network1.third_layer[k].activation;
        std::cout<<"\n";
    }
    for(int k=0; k<exit_layer_neurons; k++){
        std::cout<<network1.exit_layer[k].activation;
        std::cout<<"\n";
    }
}

double generate_random_weight(){
    return (rand()%100)/100.0;
}

network initialize_network(){
    network new_network;
    std::cout << "network initializing!" << std::endl;

    for(int i=0; i<layer_1_neurons; i++){
        new_network.entry_layer[i].activation = 1;
        std::cout << "network initializingiiiiii!" << std::endl;

        for(int j=0; j<layer_2_neurons; j++){
            new_network.layers_1_2_links[i*layer_2_neurons + j].weight = generate_random_weight();
            std::cout << "i=" << i << "j=" << j <<"i*layer_1_neurons + j = "<< i*layer_1_neurons + j << std::endl;

        }
    }
    std::cout << "network initialized1!" << std::endl;

    for(int i=0; i<layer_2_neurons; i++){
        new_network.second_layer[i].activation = 2;
        for(int j=0; j<layer_3_neurons; j++){
            new_network.layers_2_3_links[i*layer_3_neurons + j].weight = generate_random_weight();
        }
    }
    for(int i=0; i<layer_3_neurons; i++){
        new_network.third_layer[i].activation = 3;
        for(int j=0; j<exit_layer_neurons; j++){
            new_network.layer_3_exit_links[i*exit_layer_neurons + j].weight = generate_random_weight();
        }
    }
    for(int i=0; i<exit_layer_neurons; i++){
        new_network.exit_layer[i].activation = 4;
    }

    return new_network;

}

int main() {
    double random = (rand()%100)/100.0;
    std::cout << "Hello, World!" << std::endl;
    network network1 = initialize_network();
    std::cout << "network initialized!" << std::endl;

    display_network(network1);
    std::string a;
    std::cin>>a;

    return 0;
}