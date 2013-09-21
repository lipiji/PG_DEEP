//**********************
// lipiji.sdu@gmail.com
// 2013-06-25
//*********************/ 
#ifndef DEEP_H
#define DEEP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

double sample_from_gaussian(double miu, double sigma);

class Conf
{
	public:
		string f_train_x;
		string f_train_y;
		string f_test_x;
		string f_test_y;

		int epoch;
		int batch_size;
		int *hidden_layer_size;
		int cd_k;
		double learning_rate;
		int n_layers;
		int n_labels; //num of classes
        double lamda; // penalty factor
		Conf(string, string, int, int, int*, int, double, int, int, double);
		~Conf();

};

class Dataset
{
	public:
		int N;
		int n_f;
		//double **X;
		//double *Y;
		vector<vector<double> > X;
		vector<double> Y;
		int batch_index;
		Dataset(Conf);
		~Dataset();
        void reloadx(Conf, char*, vector<vector<double> >&);
};

class RBM
{
	public:
		int n_samples;
		int n_visible;
		int n_hidden;
		double **W;
		double *hbias;
		double *vbias;
        double error;

		RBM(int, int, int,  double**, double*, double*);
		~RBM();
		void train(double*, double, int);
		void activate_hidden(double*, double*, int*, int, int);
		void activate_visible(int*, double*, int*, int, int);
};
class LR
{
    public:
        int n_samples;
        int n_features;
        int n_labels;
        double lamda;
        double **W;
        double *b;
        LR(Dataset, Conf);
        LR(int, int, int, double);
        ~LR();
        void train(double*, int*, double);
        void softmax(double*);
        void predict(double*, double*);

};
class DBN
{
	public:
		int n_samples;
		int n_features;
		int n_layers;
		int n_labels;
		int *hidden_layer_size;
        double lamda;
        double alpha;
		RBM **rbm_layers;
        LR *lr_layer;
		DBN(Dataset, Conf);
		~DBN();
		void pretrain(Dataset, Conf);
		void finetune(Dataset, Conf);
		int predict(double*, double*, int);

};
#endif //DEEP_H



