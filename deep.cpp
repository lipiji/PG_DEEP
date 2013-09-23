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

double square_error(double *v1, double *v2, int size)
{
    double error = 0;
    for(int i=0; i<size; i++)
        error += pow((v1[i] - v2[i]), 2);

    return error;
}

int max_i_(double *x, int l)
{
    double max_v = x[0];
    double max_i = 0;
    for(int i=0; i<l; i++)
    {
        if(x[i] > max_v)
        {
            max_v = x[i];
            max_i = i;
        }
    }
    return max_i;
}


Conf::Conf(string ftx, string fty, int epc, int bs, int *hls, int k, double lr, int n_ly, int n_lb, double lbd)
{
    f_train_x = ftx;
    f_train_y = fty;
    epoch = epc;
    batch_size = bs;
    hidden_layer_size = hls;
    cd_k = k;
    learning_rate = lr;
    n_layers = n_ly;
    n_labels = n_lb;
    lamda = lbd;
}
Conf::~Conf(){}
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
void Dataset::reloadx(Conf conf, char* xl, vector<vector<double> >& xlayer)
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
        ifstream fin_x(xl);
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
                xlayer.push_back(x);
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
RBM::RBM(int N, int n_f, int n_h, double **w, double *hb, double *vb, double lbd, double mmt)
{
    n_samples = N;
    n_visible = n_f;
    n_hidden = n_h;
    error = 0.0;
    lamda = lbd;
    momentum = mmt;

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
void RBM::activate_hidden(double *v_prob, double *h_prob, int *h_state, int n_visible, int n_hidden)
{	
    //v->h
    if(h_prob == NULL)
        h_prob = new double[n_hidden];
    if(h_state == NULL)
        h_state = new int[n_hidden];	
    for(int i=0; i<n_hidden; i++)
    {
        double vh_prob = 0;
        for(int j=0; j<n_visible; j++)
        {
            // prob or state?
            vh_prob += v_prob[j] * W[i][j];
        }
        vh_prob += hbias[i];
        h_prob[i] = sigmoid(vh_prob);
        h_state[i] = binomial(1, h_prob[i]);
    }

}
void RBM::activate_visible(int *h_state, double *v_prob, int *v_state, int n_hidden, int n_visible)
{	
    //h->v
    if(v_prob == NULL)
        v_prob = new double[n_visible];
    if(v_state == NULL)
        v_state = new int[n_visible];	
    for(int i=0; i<n_visible; i++)
    {
        double hv_prob = 0;
        for(int j=0; j<n_hidden; j++)
        {
            hv_prob += h_state[j] * W[j][i];
        }
        hv_prob += vbias[i];
        v_prob[i] = sigmoid(hv_prob);
        v_state[i] = binomial(1, v_prob[i]);
    }	
}

void RBM::train(double *x, double gamma, int cd_k)
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
            W[i][j] += momentum * 0 + gamma * ((x[j] * pos_h_prob[i] - neg_v_prob[j] * neg_h_prob[i]) / 1 - lamda * W[i][j]); 
        }
        hbias[i] += momentum * 0 + gamma * (pos_h_prob[i] - neg_h_prob[i]) / 1;
    }
    for(int i=0; i<n_visible; i++)
    {
        vbias[i] += momentum * 0 + gamma * (x[i] - neg_v_prob[i]) / 1; // 1 is #samples
    }

    error = square_error(x, neg_v_prob, n_visible);

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
        delete[] W[i];
    delete[] W;
    delete[] hbias;
    delete[] vbias;
}

LR::LR(Dataset data, Conf conf)
{
    n_samples = data.N;
    n_features = data.n_f;
    n_labels = conf.n_labels;
    lamda = conf.lamda;

    W = new double*[n_labels];
    for(int i=0; i<n_labels; i++)
        W[i] = new double[n_features];
    b = new double[n_labels];

    for(int i=0; i<n_labels; i++)
    {
        for(int j=0; j<n_features; j++)
            W[i][j] = 0;
        b[i] = 0;
    }
}
LR::LR(int N, int n_f, int n_lb, double lbd)
{
    n_samples = N;
    n_features = n_f;
    n_labels = n_lb;
    lamda = lbd;

    W = new double*[n_labels];
    for(int i=0; i<n_labels; i++)
        W[i] = new double[n_features];
    b = new double[n_labels];

    for(int i=0; i<n_labels; i++)
    {
        for(int j=0; j<n_features; j++)
            W[i][j] = 0;
        b[i] = 0;
    }
}

LR::~LR()
{
    for(int i=0; i<n_labels; i++)
        delete[] W[i];
    delete[] W;
    delete[] b;
}
void LR::train(double *x, int *y, double gamma)
{
    double *h_theta_x = new double[n_labels];
    double *dy = new double[n_labels];

    for(int i=0; i<n_labels; i++) 
    {
        h_theta_x[i] = 0;
        for(int j=0; j<n_features; j++) 
        {
            h_theta_x[i] += W[i][j] * x[j];
        }
        h_theta_x[i] += b[i];
    }
    softmax(h_theta_x);

    for(int i=0; i<n_labels; i++) 
    {
        dy[i] = (y[i]==1 ? 1:0) - h_theta_x[i];

        for(int j=0; j<n_features; j++) 
        {
            W[i][j] -= gamma * (-1*dy[i] * x[j] + lamda * W[i][j]);
        }

        b[i] += gamma * dy[i];
    }
    delete[] h_theta_x;
    delete[] dy;
}
void LR::softmax(double *x)
{
    double sum = 0.0;
    for(int i=0; i<n_labels; i++) 
    {
        x[i] = exp(x[i]);
        sum += x[i];
    } 

    for(int i=0; i<n_labels; i++) 
        x[i] /= sum;
}
void LR::predict(double *x, double *y)
{
    for(int i=0; i<n_labels; i++) 
    {
        y[i] = 0;
        for(int j=0; j<n_features; j++) 
        {
            y[i] += W[i][j] * x[j];
        }
        y[i] += b[i];
    }
    softmax(y);
}
DBN::DBN(Dataset data, Conf conf)
{
    n_samples = data.N;
    n_features = data.n_f;
    n_layers = conf.n_layers;
    n_labels = conf.n_labels;
    lamda = conf.lamda;
    alpha = conf.learning_rate;

    rbm_layers = new RBM*[n_layers];
    for(int i=0; i<n_layers; i++)
    {
        hidden_layer_size[i] = conf.hidden_layer_size[i];
        if(i == 0)
        {
            rbm_layers[i] = new RBM(n_samples, n_features, hidden_layer_size[i], NULL, NULL, NULL, conf.lamda, 0.5);
        }
        else
        {
            rbm_layers[i] = new RBM(n_samples, hidden_layer_size[i-1], hidden_layer_size[i],  NULL, NULL, NULL, conf.lamda, 0.5);
        }
    }

    //logistic layer
    lr_layer = new LR(n_samples, hidden_layer_size[n_layers-1], n_labels, conf.lamda);

}
void DBN::pretrain(Dataset data, Conf conf)
{
    cout << "Layer-wise pre-training begin: " <<endl;

    char str[] = "./model/x_layer_";
    char x_l[256];
    vector<vector<double> > xlayer;

    double *rbm_input = NULL;
    double *pre_rbm_input = NULL;
    int pre_size;
    for(int l=0; l<n_layers; l++)
    {
        if(l > 0)
            data.reloadx(conf, x_l, xlayer);
        cout << "Layer: " << (l+1) << endl;
        for(int epoch=0; epoch<conf.epoch; epoch++)
        {
            if(epoch < 5)
                rbm_layers[l]->momentum = 0.5;
            else
                rbm_layers[l]->momentum = 0.9;

            double error = 0;
            for(int i=0; i<n_samples; i++)
            {
                if(l == 0)
                {
                    rbm_input = new double[n_features];
                    for(int f=0; f<n_features; f++)
                        rbm_input[f] = data.X[i][f];
                }
                else
                {
                    rbm_input = new double[hidden_layer_size[l-1]];
                    for(int f=0; f<hidden_layer_size[l-1]; f++)
                        rbm_input[f] = xlayer[i][f];

                }
                rbm_layers[l]->train(rbm_input, conf.learning_rate, conf.cd_k);
                error += rbm_layers[l]->error;
            }
            cout << "Layer: " << (l+1) << ", Epoch: " << epoch << ", Error: " << error << endl;
        }// end one layer

        //save the samples
        x_l[0]='\0';
        sprintf(x_l, "%s%d", str, (l+2));
        ofstream fout(x_l);
        if(l == 0)
            pre_size = n_features;
        else
            pre_size = hidden_layer_size[l-1];
        pre_rbm_input = new double[pre_size];
        for(int i=0; i<n_samples; i++)
        {
            for(int f=0; f<pre_size; f++)
            {
                if(l == 0)
                    pre_rbm_input[f] =  data.X[i][f];
                else
                    pre_rbm_input[f] =  xlayer[i][f];
            }
            rbm_input = new double[hidden_layer_size[l]];
            rbm_layers[l]->activate_hidden(pre_rbm_input, rbm_input, NULL, pre_size, hidden_layer_size[l]);
            for(int f=0; f<hidden_layer_size[l]; f++)
            {
                fout << rbm_input[f] << " ";
            }
            fout << endl;
            delete[] rbm_input;
        }
        delete[] pre_rbm_input;
        fout << flush;
        fout.close();
        if(l > 0)
            vector<vector<double> >().swap(xlayer);
    }//rbm layers end

    // logistic layer training
    data.reloadx(conf, x_l, xlayer);
    pre_size = lr_layer->n_features;
    rbm_input = new double[pre_size];

    for(int epoch=0; epoch<conf.epoch; epoch++)
    {
        for(int i=0; i<n_samples; i++)
        {
            int *train_y = new int[lr_layer->n_labels];
            for(int yi=0; yi<lr_layer->n_labels; yi++)
                train_y[yi] = 0;
            train_y[int(data.Y[i])] = 1;
            for(int f=0; f<pre_size; f++)
                rbm_input[f] = xlayer[i][f];

            //
            lr_layer->train(rbm_input, train_y, conf.learning_rate);
        }
        cout << "Layer: logistic layer, Epoch: " << epoch  << endl;
    }
    delete[] rbm_input;

}
void DBN::finetune(Dataset data, Conf conf)
{
    cout << "Fine-tuning..." << endl;
    // BP fine-tuning method
    // First, train the LR model without fine-tuning
    double *layer_input = NULL;
    double *pre_layer_input = NULL;
    int pre_size;

    for(int epoch=0; epoch<conf.epoch/2; epoch++)
    {
        //ofstream fout("./model/x.txt");
        for(int i=0; i<n_samples; i++)
        {
            vector<vector<double> > ai;
            vector<vector<double> > deltai;

            int *train_y = new int[n_labels];
            for(int yi=0; yi<n_labels; yi++)
                train_y[yi] = 0;
            train_y[int(data.Y[i])] = 1;
            // some x may be computed many times
            // which may be replaced by DP algoritham
            for(int j=0; j<n_layers; j++)
            {
                if(j == 0)
                {
                    pre_size = n_features;
                    pre_layer_input = new double[pre_size];
                    for(int f=0; f<pre_size; f++)
                        pre_layer_input[f] = data.X[i][f];
                    
                    //a0 = v0
                    vector<double> al;
                    for(int ia=0; ia<pre_size; ia++)
                        al.push_back(pre_layer_input[ia]);
                    ai.push_back(al);
                    vector<double>().swap(al);
                }
                else
                {
                    pre_size = hidden_layer_size[j-1];
                    pre_layer_input = new double[pre_size];

                    for(int f=0; f<pre_size; f++)
                        pre_layer_input[f] = layer_input[f];
                    delete[] layer_input;

                }
                layer_input = new double[hidden_layer_size[j]];
                rbm_layers[j]->activate_hidden(pre_layer_input, layer_input, NULL, pre_size, hidden_layer_size[j]);
                delete[] pre_layer_input;
                vector<double> al;
                for(int ia=0; ia<hidden_layer_size[j]; ia++)
                    al.push_back(layer_input[ia]);
                ai.push_back(al);
                vector<double>().swap(al);
            }
            ///////////
            // for(int ii=0; ii<hidden_layer_size[n_layers-1]; ii++)
            //   fout << layer_input[ii] << " ";
            //fout << int(data.Y[i]) << endl;
            /////////

            //output layer
            //in http://deeplearning.stanford.edu/wiki/index.php/Fine-tuning_Stacked_AEs
            double *pred_y = new double[lr_layer->n_labels];
            lr_layer->predict(layer_input, pred_y);
            vector<double> di0;
            for(int j=0; j<lr_layer->n_labels; j++)
                di0.push_back(-1*(train_y[j] - pred_y[j]) * pred_y[j] * (1 - pred_y[j]));
            deltai.push_back(di0);
            vector<double>().swap(di0);
            // update the parameters in LR layer
            //lr_layer->train(layer_input, train_y, conf.learning_rate);

            // hidden layer
            for(int l=n_layers; l>=0; l--)
            {
                if(l == n_layers)
                {
                    vector<double> di;
                    for(int j=0; j<lr_layer->n_features; j++)
                    {
                        double di_tmp = 0;
                        for(int f=0; f<lr_layer->n_labels; f++)
                        {
                            di_tmp += lr_layer->W[f][j] * deltai[n_layers-l][f];
                        }
                        di_tmp *= ai[l][j] * (1 -  ai[l][j]);
                        di.push_back(di_tmp);
                    }
                    deltai.push_back(di);
                    vector<double>().swap(di);
                }
                else
                {
                    vector<double> di;
                    for(int j=0; j<rbm_layers[l]->n_visible; j++)
                    {
                        double di_tmp = 0;
                        for(int f=0; f<rbm_layers[l]->n_hidden; f++)
                        {
                            di_tmp += rbm_layers[l]->W[f][j] * deltai[n_layers-l][f];
                        }
                        di_tmp *= ai[l][j] * (1 -  ai[l][j]);
                        di.push_back(di_tmp);
                    }
                    deltai.push_back(di);
                    vector<double>().swap(di);
                }
            }// end deltai
            // update hidden layer parameters
            for(int l=0; l<n_layers; l++)
            {
                for(int j=0; j<rbm_layers[l]->n_hidden; j++)
                {
                    for(int f=0; f<rbm_layers[l]->n_visible; f++)
                    {
                        rbm_layers[l]->W[j][f] -= alpha*(deltai[n_layers-l][j]*ai[l][f] + lamda * rbm_layers[l]->W[j][f]); 
                    }
                    rbm_layers[l]->hbias[j] -= alpha * deltai[n_layers-l][j];
                }
            }
            // update the parameters in LR layer
            lr_layer->train(layer_input, train_y, conf.learning_rate);
        }//end xi
        ////////////
        //fout << flush;fout.close();
        cout << "Fine-tuning epoch: " << epoch <<endl;
    }//end epoch
    cout << "Fine-tuning done." << endl;

}
int DBN::predict(double *x, double *pred_y, int true_label)
{ 
    double *layer_input = NULL;
    double *pre_layer_input = NULL;
    int pre_size;

    // some x may be computed many times
    // which may be replaced by DP algoritham
    for(int j=0; j<n_layers; j++)
    {
        if(j == 0)
        {
            pre_size = n_features;
            pre_layer_input = new double[pre_size];
            for(int f=0; f<pre_size; f++)
                pre_layer_input[f] = x[f];
        }
        else
        {
            pre_size = hidden_layer_size[j-1];
            pre_layer_input = new double[pre_size];

            for(int f=0; f<pre_size; f++)
                pre_layer_input[f] = layer_input[f];
            delete[] layer_input;

        }
        layer_input = new double[hidden_layer_size[j]];
        rbm_layers[j]->activate_hidden(pre_layer_input, layer_input, NULL, pre_size, hidden_layer_size[j]);
        delete[] pre_layer_input;

    }

    for(int i=0; i<lr_layer->n_labels; i++)
    {
        pred_y[i] = 0;
        for(int j=0; j<lr_layer->n_features; j++)
        {
            pred_y[i] += lr_layer->W[i][j] * layer_input[j];
        }
        pred_y[i] += lr_layer->b[i];
    }
    lr_layer->softmax(pred_y);
    /*
       cout <<true_label<<": ";
       for(int i=0; i<lr_layer->n_labels; i++)
       cout<<pred_y[i]<<" ";
       cout<<max_i_(pred_y, lr_layer->n_labels)<<"?????"<<true_label<<endl;
       */
    if(max_i_(pred_y, lr_layer->n_labels) == true_label)
        return 1;
    return 0;
}
DBN::~DBN()
{
    for(int i=0; i<n_layers; i++)
        delete rbm_layers[i];
    delete[] rbm_layers;
    delete lr_layer;
}
