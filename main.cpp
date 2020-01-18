#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <ctime>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;


//////////////////

const int learn_files=350;
const int test_set_size=40;
float treshhold = 0.9;
const int image_height = 30;
const int image_width = 30;
const int entry_layer_neurons = image_width * image_height;
const int hidden_layers_number = 2;
const int output_layer_neurons = 4; // exit layer neurons

const int layer_neurons_number[] = {entry_layer_neurons, 8, 8, output_layer_neurons};
const int layers_number = hidden_layers_number + 2;

const float learning_const = 0.05;
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
    return x * (1.0 - x);
}

float generate_random_weight() {
    return ((rand() % 200) / 100.0) - 1.0;
}

float generate_random_bias() {
    return generate_random_weight();
}

network initialize() {
    std::cout << "network initializing!" << std::endl;
    network new_network;
    std::cout << "layers created" << std::endl;

    //Initialize layers

    for (int i = 0; i < layers_number; i++) {
        new_network.layers[i].neurons = new neuron[layer_neurons_number[i]];
        for (int j = 0; j < layer_neurons_number[i]; j++) {
            new_network.layers[i].neurons[j].activation = 0;
            if (i > 0) {
                new_network.layers[i].neurons[j].weights = new float[layer_neurons_number[i - 1]];
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
    for (int current_layer = 0; current_layer < layers_number; current_layer++) {
        std::cout << "layer: "<< current_layer << std::endl;
        for (int j = 0; j < layer_neurons_number[current_layer]; j++) {
            std::cout << network1.layers[current_layer].neurons[j].activation << std::endl;
            if (current_layer > 0) {
                std::cout << "weights" << std::endl;
                for (int previous_layer_neuron = 0;
                    previous_layer_neuron < layer_neurons_number[current_layer - 1]; previous_layer_neuron++) {
                    std::cout << network1.layers[current_layer].neurons[j].weights[previous_layer_neuron] << std::endl;
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
    for (int i = 0; i < layer_neurons_number[layer_number - 1]; i++) {
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
    float our_treshold = treshhold;
    float max_activation = 0.0;
    int class_number = -1;
    for (int i = 0; i < output_layer_neurons; i++) {
        if (our_network.layers[layers_number - 1].neurons[i].activation > max_activation) {
            max_activation = our_network.layers[layers_number - 1].neurons[i].activation;
            if (our_network.layers[layers_number - 1].neurons[i].activation > our_treshold){
                class_number = i;
            }
        };
    }
//    std::cout << "zgodnoœæ " << max_activation << std::endl;
    return class_number;
}

void backpropagate(network &our_network, int expected_class) {


    float errors[layers_number - 1][max_neurons_in_layer]; // = new float[

    for (int i = layers_number - 1; i > 0; i--) {
        for (int j = 0; j < layer_neurons_number[i]; j++)
            errors[i][j] = 0.0;
    }

    float expected;



    float output;
    for (int i = 0; i < output_layer_neurons; i++) {
        if (expected_class == i) {
            expected = 1.0;
        } else {
            expected = 0.0;
        }
        output = our_network.layers[layers_number - 1].neurons[i].activation;
        errors[layers_number - 1][i] += (expected - output) * transfer_derivative(output);
    }

    float error;
    for (int current_layer = layers_number - 2; current_layer >= 0; current_layer--) {
        for (int i = 0; i < layer_neurons_number[current_layer]; i++) {
            error = 0.0;
            output = our_network.layers[current_layer].neurons[i].activation;
            for (int j = 0; j < layer_neurons_number[current_layer+1]; j++) {
                neuron next_layer_neuron = our_network.layers[current_layer+1].neurons[j];
                error += (next_layer_neuron.weights[i] * errors[current_layer+1][j]) *
                         transfer_derivative(output);  // "/classes?
            }
            errors[current_layer][i] = error;
        }
    }
    //Errors calculated

    //Update weights
    for (int current_layer=1; current_layer<layers_number; current_layer++){
        for (int i = 0; i<layer_neurons_number[current_layer]; i++) {
            neuron updated_neuron = our_network.layers[current_layer].neurons[i];
            float change = 0.0;
            for (int j = 0; j < layer_neurons_number[current_layer - 1]; j++) {
                change = updated_neuron.weights[j] +errors[current_layer][i] * our_network.layers[current_layer-1].neurons[j].activation * learning_const;
            updated_neuron.weights[j] = change;
            }
            updated_neuron.bias = updated_neuron.bias + errors[current_layer][i] * updated_neuron.activation * learning_const;
            our_network.layers[current_layer].neurons[i] = updated_neuron;
        }
        //std::cout<<"xd"<<std::endl;
    }
}

void load_image(float pixels[entry_layer_neurons], network &our_network) {
    //inicializuje ok
    for (int i = 0; i < entry_layer_neurons; i++) {
        our_network.layers[0].neurons[i].activation = pixels[i];
    }

//    for(int i=0; i<entry_layer_neurons; i++){
//        std::cout<<our_network.layers[0].neurons[i].activation<<std::endl;
//    }
    return;
}

void check_result(float pixels[entry_layer_neurons], network &our_network, int class_number, bool write) {
    load_image(pixels, our_network);
    for (int j = 1; j < layers_number; j++) {
        for (int i = 0; i < layer_neurons_number[j]; i++) {
            calculate_neuron_activation(our_network, j, our_network.layers[j].neurons[i]);
        }
    }
//    for (int i=0; i<hidden_layer_neurons; i++){
////        std::cout<<our_network.layers[1].neurons[i].activation<<std::endl;
//    }

//    calculate_output(our_network);
    if (write) {
        int result = choose_result(our_network);
        std::cout << "wanted " << class_number << " got " << result << std::endl;
    }
//    int result = choose_result(our_network);
//        std::cout << "wanted " << class_number << " got " << result << std::endl;
}

int get_result(float pixels[entry_layer_neurons], network &our_network, int class_number, bool write){
    load_image(pixels, our_network);
    for (int j = 1; j < layers_number; j++) {
        for (int i = 0; i < layer_neurons_number[j]; i++) {
            calculate_neuron_activation(our_network, j, our_network.layers[j].neurons[i]);
        }
    }
    return choose_result(our_network);
}

void learn(float pixels[entry_layer_neurons], network &our_network, int class_number, bool write) {
//    load_image(pixels, our_network);
    check_result(pixels, our_network, class_number, false);
    int result = choose_result(our_network);
    if (write) {
        std::cout << "wanted " << class_number << " got " << result << std::endl;
    }
    backpropagate(our_network, class_number);
}



void save_waights(network our_network){
    std::ofstream o("weights.txt");
    for (int current_layer = 1; current_layer < layers_number; current_layer++) {
        std::cout << "layer: "<< current_layer << std::endl;
        for (int j = 0; j < layer_neurons_number[current_layer]; j++) {
           // std::cout << our_network.layers[current_layer].neurons[j].activation << std::endl;
            std::cout << "weights" << std::endl;
            for (int previous_layer_neuron = 0;
                previous_layer_neuron < layer_neurons_number[current_layer - 1]; previous_layer_neuron++) {
                o<< our_network.layers[current_layer].neurons[j].weights[previous_layer_neuron] << std::endl;
            }
        }
    }
}

// for debug only
void save_waights_2(network our_network){
    std::ofstream o("weights_2.txt");
    for (int current_layer = 1; current_layer < layers_number; current_layer++) {
        std::cout << "layer: "<< current_layer << std::endl;
        for (int j = 0; j < layer_neurons_number[current_layer]; j++) {
           // std::cout << our_network.layers[current_layer].neurons[j].activation << std::endl;
            std::cout << "weights" << std::endl;
            for (int previous_layer_neuron = 0;
                previous_layer_neuron < layer_neurons_number[current_layer - 1]; previous_layer_neuron++) {
                o<< our_network.layers[current_layer].neurons[j].weights[previous_layer_neuron] << std::endl;
            }
        }
    }
}
// debug

void load_weights(network our_network){
    //std::cout<<"XDDDDDD"<<std::endl;

    std::ifstream infile("weights.txt");
    std::string line;
    float weight;
    for (int current_layer = 1; current_layer < layers_number; current_layer++) {
        //std::cout << "layer: "<< current_layer << std::endl;
        for (int j = 0; j < layer_neurons_number[current_layer]; j++) {
            std::cout << our_network.layers[current_layer].neurons[j].activation << std::endl;
          //  std::cout << "loading weights" << std::endl;
            for (int previous_layer_neuron = 0;
                previous_layer_neuron < layer_neurons_number[current_layer - 1]; previous_layer_neuron++) {
                std::getline(infile, line);

             //   our_network.layers[current_layer].neurons[j].weights[previous_layer_neuron] = std::stof(line);
                infile>>our_network.layers[current_layer].neurons[j].weights[previous_layer_neuron];
               // std::cout<<our_network.layers[current_layer].neurons[j].weights[previous_layer_neuron]<<std::endl;
                //std::cout<<"loadedXD"<<std::endl;
            }
        }
    }
}


float * get_note_from_txt(std::string filename, int &noteclass){
    float * arr=new float[entry_layer_neurons];
    std::ifstream infile(filename);
    std::string line;
    infile>>noteclass;
    for(int i=0; i<entry_layer_neurons; i++){
        int data=0;
        infile>>data;
        arr[i]=(float)data;
   //     std::cout<<data<<std::endl;
    }
    return arr;
}

void learn_from_random_file(network &our_network){
    int file_number = rand()%learn_files;
    std::stringstream ss;
    ss << file_number;
    std::string filename = ss.str();
 //   std::string filename =  std::to_string(file_number);
    filename = filename + ".txt";
    int noteclass = 0;
    float * arr = get_note_from_txt("tekstfiles/" + filename, noteclass);
    learn(arr, our_network, noteclass, false);
}


void check_network(network &our_network){
    int got_right=0;
    for(int file_number=0; file_number<test_set_size; file_number++){
        std::stringstream ss;
        ss << file_number;
        std::string filename = ss.str();
        filename = filename + ".txt";
        int noteclass = 0;
        float * arr = get_note_from_txt("test_batch/" + filename, noteclass);
        std::cout<<"class wanted"<<noteclass<<std::endl;
        int res = get_result(arr, our_network, noteclass, true);
        if(res==noteclass){
            got_right++;
        }
    }
    float max_right = test_set_size/1.0;
    float percentage = got_right/max_right;
    std::cout<<"accuracy: "<<percentage<<std::endl;

}


void train_network(network our_network, int amount){
    for(int i=0; i<amount; i++){
        learn_from_random_file(our_network);
    }
}


//////////

vector<Rect> sort_tones(vector <Rect> boundRect){
    Rect swaprect;
    for(int i=1;i<boundRect.size();i++){
        int j=i;
        while(j>0 && boundRect[j].x<boundRect[j-1].x){
            swaprect=boundRect[j-1];
            boundRect[j-1]=boundRect[j];
            boundRect[j]=swaprect;
            j--;
        }
    }
    return boundRect;
}


int get_note_height(int note_height, int image_height){
    float space=(float)image_height/32;
    int rev_height=image_height-note_height;
    float space_number=((float)rev_height)/space;
    //cout<<image_height<<" "<<((int)space_number1)/2<<endl;
    return ((int)round(space_number)+1)/2;
}

vector<vector<int>> pixels_and_pictures30(620*450,vector<int>(0));

vector<vector<int>> pixels_and_pictures50(620*450,vector<int>(0));

vector<vector<int>> pixels_and_pictures70(620*450,vector<int>(0));

int main(int argc, char** argv)
{

    RNG rng(12345);

    int minm=120,maxm=40,minc=150,maxc=300;
    VideoCapture cap(0);
    if(!cap.isOpened())return -1;
    Mat frame,gray_src,abc,frame2,dst,cdst,cropped;
    namedWindow("input image", 1);
    namedWindow("filter lines", 1);
    namedWindow("treshold image", 1);
    createTrackbar("min", "filter lines", &minm, 400);
    createTrackbar("max", "filter lines", &maxm, 500);
    createTrackbar("minc", "filter lines", &minc, 400);
    createTrackbar("maxc", "filter lines", &maxc, 500);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Vec4i> lines;
    int minx,maxx,miny,maxy;

    for(;;)
    {
        cap >> frame;
        cap >> frame2;
        minx=1000000;
        maxx=0;
        miny=100000;
        maxy=0;
        cvtColor( frame, gray_src, CV_BGR2GRAY );
        Canny(frame, dst, minc, maxc, 3);
        cvtColor(dst, cdst, CV_GRAY2BGR);
        blur(dst, dst, Size(5 , 5));
        HoughLinesP(dst, lines, 1, CV_PI/180, minm, maxm, 000 );
        for( size_t i = 0; i < lines.size(); i++ )
          {
            Vec4i l = lines[i];
            maxx=max(maxx,max(l[0],l[2]));
            maxy=max(maxy,max(l[1],l[3]));
            minx=min(minx,min(l[0],l[2]));
            miny=min(miny,min(l[1],l[3]));
            line( gray_src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 1, CV_AA);
          }
        cout<< "RECTANGLE: " << minx<<" "<<maxx<<" "<<miny<<" "<<maxy<<endl;
        if( minx<maxx && miny<maxy){
            Rect rect(minx, miny, maxx-minx, maxy-miny );
            cropped = frame(rect);
            rectangle(gray_src, rect, cv::Scalar(255, 255, 255));
            imwrite("cropped.jpg", cropped);
        }
        imshow("input image",frame);
        imwrite("detected.jpg", frame2);
        imshow("treshold image", gray_src);

        if(waitKey(400) >= 0) break;   // you can increase delay to 2 seconds here
}

    //int bordery = (maxy-miny)/2;
    //Rect rect(minx-15, miny-bordery, maxx-minx+30, maxy-miny+2*bordery );
    cropped = frame2;
    imwrite("cropped.jpg", cropped);


    int tres=127,blurr=1,dilatee=1,erodee=1;
    int img_height;
    for(;;){
        Mat img,source;
        img = imread("cropped.jpg", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
        namedWindow( "linefound", WINDOW_NORMAL );// Create a window for display.
        resizeWindow("linefound",300,300);
        namedWindow( "Display Window1", WINDOW_AUTOSIZE );// Create a window for display.
        createTrackbar("treshold", "linefound", &tres, 200);
        createTrackbar("blur", "linefound", &blurr, 8);
        createTrackbar("dilate", "linefound", &dilatee, 8);
        createTrackbar("erode", "linefound", &erodee, 8);
        threshold( img, img, tres, 255, 1);
        blur(img, img, cv::Size(blurr,blurr));
        imshow( "Display window", img );                   // Show our image inside it.
        imwrite("img.jpg", img);
        source=img.clone();
        cv::Mat m = img.clone();
        cv::threshold(m, m, tres, 255,CV_THRESH_BINARY_INV);
        cv::erode(m, m, cv::Mat(),cv::Point(-1,-1),erodee);
        cv::dilate(m, m, cv::Mat(),cv::Point(-1,-1),dilatee);
         vector< std::vector<cv::Point> > contours;
         vector<Vec4i> hierarchy;
         m=255-m;
         findContours( m, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
          vector<vector<Point> > contours_poly( contours.size() );
          vector<Rect> boundRect( contours.size() );
          cout<<contours.size()<<endl;
          for( int i = 0; i < contours.size(); i++ ){
                approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
                boundRect[i] = boundingRect( Mat(contours_poly[i]) );
             }

          for( int i = 0; i< contours.size(); i++ )
             {
               Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
               rectangle( img, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
             }
            img_height=img.rows;
            namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
            imshow( "Contours", img );


      if(waitKey(60) >= 0){
        /*    cout<<img.rows<<endl;
        boundRect=sort_tones(boundRect);
        Mat objectframe;
        threshold( source, source, tres, 255, 1);
        for(int i=0;i<boundRect.size();i++){
            objectframe = source(Rect(boundRect[i].x,boundRect[i].y,boundRect[i].width,boundRect[i].height)).clone();
            string name =to_string(i+150)+".jpg";
            Size objectsize(30,30);
            resize(objectframe,objectframe,objectsize);
            string txtname =to_string(i+150)+".txt";
            ofstream outfile (txtname);
            outfile<<2<<" ";
            for(int j=0;j<30;j++){
                for(int k=0;k<30;k++){
                    int val=(int)(objectframe.at<uchar>(j,k));
                    if(val>0)val=0;
                    else val=1;
                    outfile<<val<<" ";
                   // cout<<val<<" ";// ASCII ART
                }
                //cout<<endl; //ASCIII ART
            }
            cout<<endl;
            cout<<endl;
            outfile.close();
            imwrite(name,objectframe);
            cout<<name<<" "<< boundRect[i].x<<" "<<boundRect[i].y+boundRect[i].height<<" number"<<get_note_height(boundRect[i].y+boundRect[i].height,img.rows)<<endl;

        }*/
        int k=0;
        Mat objectframe;
        threshold( source, source, tres, 255, 1);
        for(int i=0;i<600;i++){
            for(int j=0;j<440;j++){
                objectframe = source(Rect(i*5,j*5,70,70)).clone();
                //string name =to_string(k)+".jpg";
                Size objectsize(30,30);
                resize(objectframe,objectframe,objectsize);
                //string txtname =to_string(k)+".txt";
                //ofstream outfile (txtname);
                for(int j=0;j<30;j++){
                    for(int l=0;l<30;l++){
                        int val=(int)(objectframe.at<uchar>(j,l));
                        if(val>0)val=0;
                        else val=1;
                        //outfile<<val<<" ";
                        pixels_and_pictures70[k].push_back(val);
                    }
                }
                //outfile.close();
                //imwrite(name,objectframe);
                //cout<<k<<" "<<pixels_and_pictures[k].size()<<endl;
                k++;
            }
        }
        k=0;
        for(int i=0;i<116;i++){
            for(int j=0;j<65;j++){
                objectframe = source(Rect(i*5,j*5,50,50)).clone();
                //string name =to_string(k)+".jpg";
                Size objectsize(30,30);
                resize(objectframe,objectframe,objectsize);
                //string txtname =to_string(k)+".txt";
                //ofstream outfile (txtname);
                for(int j=0;j<30;j++){
                    for(int l=0;l<30;l++){
                        int val=(int)(objectframe.at<uchar>(j,l));
                        if(val>0)val=0;
                        else val=1;
                        //outfile<<val<<" ";
                        pixels_and_pictures50[k].push_back(val);
                    }
                }
                //outfile.close();
                //imwrite(name,objectframe);
                //cout<<k<<" "<<pixels_and_pictures[k].size()<<endl;
                k++;
            }
        }
        k=0;
        for(int i=0;i<122;i++){
            for(int j=0;j<90;j++){
                objectframe = source(Rect(i*5,j*5,30,30)).clone();
                //string name =to_string(k)+".jpg";
                Size objectsize(30,30);
                resize(objectframe,objectframe,objectsize);
                //string txtname =to_string(k)+".txt";
                //ofstream outfile (txtname);
                for(int j=0;j<30;j++){
                    for(int l=0;l<30;l++){
                        int val=(int)(objectframe.at<uchar>(j,l));
                        if(val>0)val=0;
                        else val=1;
                        //outfile<<val<<" ";
                        pixels_and_pictures30[k].push_back(val);
                    }
                }
                //outfile.close();
                //imwrite(name,objectframe);
                //cout<<k<<" "<<pixels_and_pictures[k].size()<<endl;
                k++;
            }
        }
        break;
      }
    }
/////////  AI
    srand(time(0));
    float random_arr[entry_layer_neurons];
    float random_arr2[entry_layer_neurons];
    float random_arr3[entry_layer_neurons];
    float random_arr22[entry_layer_neurons];
    float random_arr33[entry_layer_neurons];
    float random_arr35[entry_layer_neurons];


    for (int i = 0; i < entry_layer_neurons; i++) {
        random_arr2[i] = 2;
        random_arr3[i] = -1;
        random_arr22[i] = 1.8;
        random_arr33[i] = -1.2;
        random_arr35[i] = -1.1;

    }
    network our_network = initialize();
    //save_waights_2(our_network);

    load_weights(our_network);


//    for (int i = 0; i < 1000; i++) {
////        learn(random_arr, our_network, 1, false);
//        if(i%10000==0){
//            std::cout<<"10000"<<std::endl;
//        }
//        learn(random_arr3, our_network, 1, true);
//        learn(random_arr2, our_network, 0, false);
//        learn(random_arr33, our_network, 1, false);
//        learn(random_arr22, our_network, 0, false);
//    }
//    check_result(random_arr35, our_network, 1, true);

//    display_network(our_network);
//    save_waights(our_network);
/*
    train_network(our_network, 20000);
    learn_from_random_file(our_network);
    int note_class = 66;
    float * arr = get_note_from_txt("4.txt", note_class);
    std::cout<<"ENDOFLEARNING "<<std::endl;

    check_result(arr, our_network, 1, true);
    //save_waights(our_network);
    check_network(our_network);
*/
/////
    char d;
    float arrar[900];
    for(int i=0;i<900;i++)arrar[i]=0.0;
    check_result(arrar,our_network,-1,true);
    int p=0;
    for(int i=0;i<pixels_and_pictures30.size();i++){
        if(pixels_and_pictures30[i].size()!=900)continue;
        float arr[900];
        for(int j=0;j<900;j++)
            arr[j] = (float)pixels_and_pictures30[i].at(j);
        int a = get_result(arr,our_network,4,0);
        if(a!=-1){
            cout<<a<<" "<<i<<" "<<our_network.layers[layers_number-1].neurons[0].activation<<" "<<our_network.layers[layers_number-1].neurons[1].activation<<" "<<our_network.layers[layers_number-1].neurons[2].activation<<" "<<our_network.layers[layers_number-1].neurons[3].activation<<endl;
            for(int k=0;k<30;k++){
                for(int l=0;l<30;l++)
                    cout<<pixels_and_pictures30[i].at(30*k+l)<<" ";
                cout<<endl;
            }
            std::cin>>d;
            p++;
            //system("cls");
        }
    }
    cout<<"found:"<<p<<endl;
    p=0;
    for(int i=0;i<pixels_and_pictures50.size();i++){
        if(pixels_and_pictures50[i].size()!=900)continue;
        float arr[900];
        for(int j=0;j<900;j++)
            arr[j] = (float)pixels_and_pictures50[i].at(j);
        int a = get_result(arr,our_network,4,0);
        if(a!=-1){
            cout<<a<<" "<<i<<" "<<our_network.layers[layers_number-1].neurons[0].activation<<" "<<our_network.layers[layers_number-1].neurons[1].activation<<" "<<our_network.layers[layers_number-1].neurons[2].activation<<" "<<our_network.layers[layers_number-1].neurons[3].activation<<endl;for(int k=0;k<30;k++){
                for(int l=0;l<30;l++)
                    cout<<pixels_and_pictures50[i].at(30*k+l)<<" ";
                cout<<endl;
            }
            std::cin>>d;
            p++;
            //system("cls");
        }
    }
    cout<<"found:"<<p<<endl;
    p=0;
    for(int i=0;i<pixels_and_pictures70.size();i++){
        if(pixels_and_pictures70[i].size()!=900)continue;
        float arr[900];
        for(int j=0;j<900;j++)
            arr[j] = (float)pixels_and_pictures70[i].at(j);
        int a = get_result(arr,our_network,4,0);
        if(a!=-1){
            cout<<a<<" "<<i<<" "<<our_network.layers[layers_number-1].neurons[0].activation<<" "<<our_network.layers[layers_number-1].neurons[1].activation<<" "<<our_network.layers[layers_number-1].neurons[2].activation<<" "<<our_network.layers[layers_number-1].neurons[3].activation<<endl;for(int k=0;k<30;k++){
                for(int l=0;l<30;l++)
                    cout<<pixels_and_pictures70[i].at(30*k+l)<<" ";
                cout<<endl;
            }
            std::cin>>d;
            p++;
            //system("cls");
        }
    }
    cout<<"found:"<<p<<endl;
 return 0;
}
