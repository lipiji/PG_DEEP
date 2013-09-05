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
		int n_h;
		int cd_k;

		Conf(string, string, int, int, int, int);

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

		RBM(Conf, Dataset, double**, double*, double*);
		~RBM();
		void train(vector<double>, double, int);
};


#endif //DEEP_H



