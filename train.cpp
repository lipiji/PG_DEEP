//**********************
// lipiji.sdu@gmail.com
// 2013-06-25
//*********************/ 
#include "deep.h"
int main(int argc, const char *argv[])
{
	string ftx = "./data/test_x.txt";
	string fty = "./data/test_y.txt";
	int epoch = 100;
	int batch_size = 0;
	int n_hidden = 100;
	double gamma = 0.1; // learning rate
	int k = 1; //Contrastive Divergence k

	Conf conf(ftx, fty, epoch, batch_size, n_hidden, k);
	Dataset data(conf);
	RBM rbm(conf, data, NULL, NULL, NULL);

	for(int i=0; i<epoch; i++)
	{
		cout << "epoch: " << i << endl;
		for(int j=0; j<data.N; j++)
			rbm.train(data.X[j], gamma, k);

	}
	return 0;
}
