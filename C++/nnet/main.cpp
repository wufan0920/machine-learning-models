#include <iostream>
#include <fstream>
#include <Dense>
#include <string>
#include <vector>
#include <math.h>
#include <time.h>
#include "stdlib.h"
#include "stdio.h"
#include <minfunc.h>
#include <util.h>

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

void sampling_image(MatrixXd& data, MatrixXd& img, int patch_num, int patch_size)
{
    int xPos;
    int yPos;
    MatrixXd tmp;
    int row_limit = img.rows();
    int col_limit = img.cols();

    srand(time(NULL));

    for(int index = 0; index < patch_num; index++)
    {
        xPos = rand()%(row_limit - patch_size);
        yPos = rand()%(col_limit - patch_size);
        tmp = img.block(xPos, yPos, patch_size, patch_size);
        tmp.resize(patch_size*patch_size,1);
        data.col(index) = tmp;
    }

    return;
}

void get_training_data(MatrixXd& data)
{
    const char* prefix = "../dataset/imageData/image";
    char filename[64];
    MatrixXd img(512,512);
    MatrixXd slice(64,1000);

    for(int index = 0; index <= 9; index++)
    {
        sprintf(filename, "%s%d", prefix, index);
        ifstream indata(filename);
        load_data(img, indata);
        sampling_image(slice, img, 1000, 8);
        data.middleCols(index*1000, 1000) = slice;
    }
    
    return;
}

void normalization(MatrixXd& data)
{
    int row = data.rows();
    int col = data.cols();
    MatrixXd stds(row, col);
    VectorXd means = data.colwise().mean();
    double deviation = 3 * stDeviation(data);

    stds.fill(deviation);

    data = colwise_minus(data, means);

    data = data.cwiseMin(stds).cwiseMax(-stds);
    data /= deviation;
    data = (data.array() + 1) * 0.4;
    data = data.array() + 0.1;
    return;
}

MatrixXd flatten_params(vector<MatrixXd>& params)
{
    //TODO:iterator mode
    //i.e:vector<MatrixXd>::iterator iter=ivec.begin;iter!=ivec.end;iter++
    int result_size = 0;
    int index = 0;
    vector<int> param_size;

    for(int i = 0; i < params.size(); i++)
    {
        int tmp = params[i].rows()*params[i].cols();
        result_size += tmp;
        param_size.push_back(tmp);
    }

    MatrixXd result(result_size, 1);

    for(int i = 0; i < params.size(); i++)
    {
        int size = param_size[i];
        params[i].resize(size, 1);
        result.middleRows(index, size) = params[i];
        index += size;
    }
    
    return result;
}

MatrixXd initialize_parameters(int hidden_size, int visible_size)
{
    vector<MatrixXd> params;
    double r = sqrt(6)/sqrt(hidden_size + visible_size + 1);
    MatrixXd W1 = MatrixXd::Random(hidden_size, visible_size) * 2 * r;
    MatrixXd W2 = MatrixXd::Random(visible_size, hidden_size) * 2 * r;
    MatrixXd b1 = MatrixXd::Zero(hidden_size, 1);
    MatrixXd b2 = MatrixXd::Zero(visible_size, 1);
    W1 = W1.array() - r;
    W2 = W2.array() - r;
    params.push_back(W1);
    params.push_back(W2);
    params.push_back(b1);
    params.push_back(b2);
    return flatten_params(params);
}

MatrixXd NeuralNetworkGradient::compute_grad(MatrixXd& data, MatrixXd& label, MatrixXd& weight)
{
	float lambda = this->lambda;
	float sparsity_param = this->sparsity_param;
	float beta = this->beta;
	int visible_size = this->visible_size;
	int hidden_size = this->hidden_size;

    int index = 0;
    int size = hidden_size*visible_size;
    int num = data.cols();

    MatrixXd W1,W2,b1,b2,z2,a2,z3,a3;
    MatrixXd W1grad,W2grad,b1grad,b2grad,rho;
    MatrixXd d3, d2, dsparse;
    vector<MatrixXd> grad;
    
    W1 = weight.middleRows(0, size);
    W1.resize(hidden_size, visible_size);
    index += size; 

    W2 = weight.middleRows(index, size);
    W2.resize(visible_size, hidden_size);
    index += size; 

    b1 = weight.middleRows(index, hidden_size);
    b1.resize(hidden_size, 1);
    index += hidden_size; 

    b2 = weight.middleRows(index, visible_size);
    b2.resize(visible_size, 1);

    //step1 compute activation for each layer
    z2 = rowwise_add((W1*data), b1);
    a2 = sigmoid(z2);
    z3 = rowwise_add((W2*a2), b2);
    a3 = sigmoid(z3);
    rho = (1.0/num) * a2.rowwise().sum();

    //step2 compute grad for each layer using back propagation
    d3 = (a3 - label).cwiseProduct(sigInv(z3));
    dsparse = beta*(rho.array().inverse() * (-sparsity_param) + (1-rho.array()).array().inverse()*(1-sparsity_param) );
    d2 = rowwise_add((W2.transpose()*d3), dsparse).cwiseProduct(sigInv(z2));

    //step3 compute gradient
    W1grad = (1.0/num) * d2*data.transpose() + lambda*W1;
    W2grad = (1.0/num) * d3*a2.transpose() + lambda*W2;
    b1grad = (1.0/num) * d2.rowwise().sum();
    b2grad = (1.0/num) * d3.rowwise().sum();
    
    grad.push_back(W1grad);
    grad.push_back(W2grad);
    grad.push_back(b1grad);
    grad.push_back(b2grad);

    return flatten_params(grad);
}

int main(int argc, char** argv) 
{
    time_t start, end;
    int visible_size = 8*8;
    int hidden_size = 25;
    MatrixXd data(64,10000);
    get_training_data(data);
    normalization(data);
    MatrixXd weight = initialize_parameters(hidden_size, visible_size);

    NeuralNetworkGradient gradfunc(0.0001,0.01,3,64,25);
    MinfuncLBFGS minfunc(400, 0.3, 100);
    //cout<<weight<<endl;
    //cout<<data(0,1)<<endl;

    start = time(NULL);
    minfunc.optimize(data, data, weight, gradfunc);
    end = time(NULL);
    cout<<"time cost: "<<end-start<<endl;
    //test output
    ofstream out("out.txt");
	cout.rdbuf(out.rdbuf());
    cout<<weight.topRows(25*64)<<endl;

    return 0;
}
