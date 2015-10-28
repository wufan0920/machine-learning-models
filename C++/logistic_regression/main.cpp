#include <iostream>
#include <fstream>
#include <Dense>
#include <string>
#include <vector>
#include <math.h>
#include "stdlib.h"
#include <minfunc.h>
#include <util.h>

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

MatrixXd NewtonGradient::compute_grad(MatrixXd& data, MatrixXd& label, MatrixXd& weight)
{
    MatrixXd score, h, grad;
    int n = data.rows();
    score = data * weight;
    h = sigmoid(score);
    grad = (1.0/n) * (data.transpose() * (h - label));
    return grad;
}

int main(int argc, char** argv) 
{
    ifstream indata("../dataset/ex4Data/ex4x.dat");
    ifstream inlabel("../dataset/ex4Data/ex4y.dat");
    MatrixXd data(80,2);
    MatrixXd data_bias(80,3);
    MatrixXd label(80,1);
    MatrixXd weight(3,1);
    MinfuncNewton minfunc(7);
    NewtonGradient gradfunc;
    weight = MatrixXd::Zero(3,1);

    load_data(data,indata);
    load_data(label,inlabel);
    add_bias(&data_bias, &data);

    minfunc.optimize(data_bias, label, weight, gradfunc);

    cout<<weight<<endl;

	return 0;
}
