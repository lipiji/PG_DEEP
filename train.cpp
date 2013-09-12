//**********************
// lipiji.sdu@gmail.com
// 2013-06-25
//*********************/ 
#include "deep.h"
int main(int argc, const char *argv[])
{
	string ftx = "./data/test_x.txt";
	string fty = "./data/test_y.txt";
	int epoch = 10;
	int batch_size = 0;
	double gamma = 0.1; // learning rate
	int k = 1; //Contrastive Divergence k

	int hls[] = {400, 400, 100};
	int n_layers = sizeof(hls) / sizeof(hls[0]);
	int n_lables = 10;

	Conf conf(ftx, fty, epoch, batch_size, hls, k, gamma, n_layers, n_lables);
	Dataset data(conf);
	
	/* test rbm
	RBM rbm(data.N, data.n_f, conf.n_h, NULL, NULL, NULL);

	for(int i=0; i<epoch; i++)
	{
		cout << "epoch: " << i << endl;
		for(int j=0; j<data.N; j++)
			rbm.train(data.X[j], gamma, k);

		ofstream fout("./model/W.txt");
		for(int j=0; j<rbm.n_visible; j++)
		{
			for(int l=0; l<rbm.n_hidden; l++)
			{
				fout << rbm.W[l][j] << " ";
			}
			fout << endl;
		}
		fout << flush;
		fout.close();


	}*/
	DBN dbn(data, conf);
	dbn.pretrain(data, conf);
	for(int i=0; i<n_layers; i++)
	{
		char str[] = "./model/W";
		char W_l[128];
		sprintf(W_l, "%s%d", str, (i+1));
		
		ofstream fout(W_l);
		for(int j=0; j<dbn.rbm_layers[i]->n_visible; j++)
		{
			for(int l=0; l<dbn.rbm_layers[i]->n_hidden; l++)
			{
				fout << dbn.rbm_layers[i]->W[l][j] << " ";
			}
			fout << endl;
		}
		fout << flush;
		fout.close();

	}
	return 0;
}
