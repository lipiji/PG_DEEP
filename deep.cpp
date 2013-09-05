//**********************
// lipiji.sdu@gmail.com
// 2013-06-25
//*********************/ 
#include "deep.h"

double sample_from_gaussian(double miu, double sigma)
{
	//std::default_random_engine generator;
	//std::normal_distribution<double> distribution(miu, sigma);
	//reutrn distribution(generator);

	static double V1, V2, S;
	static int phase = 0;
	double X;

	if (phase == 0)
	{
		do
		{
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X * sigma + miu;
}

double sigmoid(double x) 
{
	return 1.0 / (1.0 + exp(-x));
}

int binomial(int n, double p) 
{
	if(p < 0 || p > 1) return 0;

	int c = 0;
	double r;

	for(int i=0; i<n; i++) {
		r = rand() / (RAND_MAX + 1.0);
		if (r < p) c++;
	}

	return c;
}

Conf::Conf(string ftx, string fty, int epc=0, int bs=0, int nh=0, int k=1)
{
	f_train_x = ftx;
	f_train_y = fty;
	epoch = epc;
	batch_size = bs;
	n_h = nh;
	cd_k = k;
}
Dataset::Dataset(Conf conf)
{

	N = 0;
	n_f = 0;
	batch_index = 0;

	if(conf.batch_size >= 0)
	{
		int Nx = 0;
		int Ny = 0;
		int Nf = 0; // dim of x

		//read the label file
		ifstream fin_y(conf.f_train_y.c_str());
		if(!fin_y) 
		{   
			cout << "Error opening " << conf.f_train_y << " for input" << endl;
			exit(-1);
		}
		else
		{
			string s;
			while(getline(fin_y, s))
			{	
				Y.push_back(atof(s.c_str()));
				if(conf.batch_size > 0 && Ny >= conf.batch_size)
					break;
				++Ny;
			}
		}
		fin_y.close();

		// read the x file
		ifstream fin_x(conf.f_train_x.c_str());
		if(!fin_x) 
		{   
			cout << "Error opening " << conf.f_train_x << " for input" << endl;
			exit(-1);
		}
		else
		{
			string s;
			while(getline(fin_x, s))
			{	
				Nf = 0;
				vector<double> x;
				const char *split = " \t";

				char *line=new char[s.size()+1];
				strcpy(line, s.c_str());
				char *p = strtok(line, split);
				x.push_back(atof(p));

				Nf = 1;
				while(1)
				{
					p = strtok(NULL, split);
					if(p == NULL || *p == '\n')
						break;
					x.push_back(atof(p));
					++Nf;
				}
				X.push_back(x);
				n_f = Nf;

				vector<double>().swap(x);
				if(conf.batch_size > 0 && Nx >= conf.batch_size)
					break;
				++Nx;
			}
		}
		fin_x.close();

		if(Nx == Ny)
		{
			N = Nx;
			n_f = Nf;
		}
		else
		{
			cout << "Dataset error: size(x) != size(y)." << endl;
			exit(-1);
		}

		cout << "Data loaded: size=" << N <<", dim=" << n_f << endl;
	}

}
Dataset::~Dataset()
{
	vector<vector<double> >().swap(X);
	vector<double>().swap(Y);
}
RBM::RBM(Conf conf, Dataset ds, double **w, double *hb, double *vb)
{
	n_samples = ds.N;
	n_visible = ds.n_f;
	n_hidden = conf.n_h;

	if(w == NULL)
	{
		W = new double*[n_hidden];
		for(int i=0; i<n_hidden; i++)
			W[i] = new double[n_visible];
		for(int i=0; i<n_hidden; i++)
			for(int j=0; j<n_visible; j++)
				W[i][j] = sample_from_gaussian(0, 0.01);
	}
	else
	{
		W = w;
	}

	if(hb == NULL)
	{
		hbias = new double[n_hidden];
		for(int i=0; i<n_hidden; i++)
			hbias[i] = 0;
	}
	else
		hbias = hb;

	if(vb == NULL)
	{
		vbias = new double[n_visible];
		for(int i=0; i<n_visible; i++)
			vbias[i] = 0;
	}
	else
		vbias = vb;


}
void RBM::train(vector<double> x, double gamma, int cd_k)
{
	double *pos_h_prob = new double[n_hidden];
	int *pos_h_state = new int[n_hidden];
	double *neg_v_prob = new double[n_visible];
	int *neg_v_state = new int[n_visible];
	double *neg_h_prob = new double[n_hidden];
	int *neg_h_state = new int[n_hidden];
	// use prob or state?
	// many tricks in 
	// A Practical Guide to Training Restricted Boltzmann Machines
	
	// postive phase
	for(int i=0; i<n_hidden; i++)
	{
		double h_prob = 0;
		for(int j=0; j<n_visible; j++)
		{
			h_prob += x[j] * W[i][j];
		}
		h_prob += hbias[i];

		pos_h_prob[i] = sigmoid(h_prob);
		pos_h_state[i] = binomial(1, pos_h_prob[i]);
	}
	// negative phase
	for(int k=1; k<=cd_k; k++)
	{
		//h0->v1
		for(int i=0; i<n_visible; i++)
		{
			double v_prob = 0;
			for(int j=0; j<n_hidden; j++)
			{
				v_prob += pos_h_state[j] * W[j][i];
			}
			v_prob += vbias[i];
			neg_v_prob[i] = sigmoid(v_prob);
			neg_v_state[i] = binomial(1, neg_v_prob[i]);
		}	
		//v1->h1	
		for(int i=0; i<n_hidden; i++)
		{
			double h_prob = 0;
			for(int j=0; j<n_visible; j++)
			{
				// prob or state?
				h_prob += neg_v_prob[j] * W[i][j];
			}
			h_prob += hbias[i];

			neg_h_prob[i] = sigmoid(h_prob);
			neg_h_state[i] = binomial(1, neg_h_prob[i]);
		}

	}
	// update parameters
	for(int i=0; i<n_hidden; i++)
	{
		for(int j=0; j<n_visible; j++)
		{
			W[i][j] += gamma * (x[j] * pos_h_prob[i] - neg_v_prob[j] * neg_h_prob[i]) / n_samples; 
		}
		hbias[i] += gamma * (pos_h_prob[i] - neg_h_prob[i]) / n_samples;
	}
	for(int i=0; i<n_visible; i++)
	{
		vbias[i] += gamma * (x[i] - neg_v_prob[i]) / n_samples;
	}



	delete[] pos_h_prob;
	delete[] pos_h_state;
	delete[] neg_v_prob;
	delete[] neg_v_state;
	delete[] neg_h_prob;
	delete[] neg_h_state;
	
}
RBM::~RBM()
{
	for(int i=0; i < n_hidden; i++)
		delete W[i];
	delete[] W;
	delete[] hbias;
	delete[] vbias;
}


