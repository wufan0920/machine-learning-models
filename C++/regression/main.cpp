#include <iostream>
#include <fstream>
#include <Dense>
#include <string>
#include <vector>
#include <math.h>
#include "stdlib.h"
#include <minfunc.h>
#include <util.h>
#include <gradient.h>

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

void analytic_solution(MatrixXd& data, MatrixXd& label, MatrixXd& weight)
{
    weight = (data.transpose() * data).inverse() * data.transpose() * label;
    return;
}

void normalization(MatrixXd& data)
{
    int col = data.cols();
    int row = data.rows();

    VectorXd means = data.colwise().mean();
    VectorXd deviation = stDeviation(data,0);

    data = colwise_minus(data, means)/deviation;
    return;
}

void regression()
{
    ifstream indata("../dataset/ex2Data/ex2x.dat");
    ifstream inlabel("../dataset/ex2Data/ex2y.dat");
    MatrixXd data(50,1);
    MatrixXd data_bias(50,2);
    MatrixXd label(50,1);
    MatrixXd weight(2,1);
    MinfuncDescent minfunc(1500, 0.07);
    RegressionGradient gradfunc; 

    weight = MatrixXd::Zero(2,1);

    load_data(data,indata);
    load_data(label,inlabel);
    add_bias(&data_bias, &data);
    //analytic_solution(data_bias, label, weight);
    //gradient_descent(data_bias, label, weight);
    minfunc.optimize(data_bias, label, weight, gradfunc);
    cout<<data_bias*weight<<endl;
    return;
}

void regression_multi()
{
    ifstream indata("../dataset/ex3Data/ex3x.dat");
    ifstream inlabel("../dataset/ex3Data/ex3y.dat");
    MatrixXd data(47,2);
    MatrixXd data_bias(47,3);
    MatrixXd label(47,1);
    MatrixXd weight(3,1);
    MinfuncDescent minfunc(100, 1.27);
    RegressionGradient gradfunc; 

    weight = MatrixXd::Zero(3,1);

    load_data(data,indata);
    load_data(label,inlabel);

    MatrixXd test_data(1,3);
    //test_data<<1650,3,1;
    test_data << (1650-data.colwise().mean()(0))/stDeviation(data, 0)(0),\
                  (3-data.colwise().mean()(1))/stDeviation(data, 0)(1),1;

    normalization(data);
    add_bias(&data_bias, &data);
    //analytic_solution(data_bias, label, weight);
    minfunc.optimize(data_bias, label, weight, gradfunc);

    //test
    cout<<test_data<<endl;
    cout<<test_data*weight<<endl;
    return;
}

MatrixXd RegressionGradient::compute_grad(MatrixXd& data, MatrixXd& label, MatrixXd& weight)
{
    int n = data.rows();
    MatrixXd grad = (1.0/n) * (data.transpose() * ((data*weight) - label));
    return grad;
}

int main(int argc, char** argv) 
{
    //regression();
    regression_multi();

	return 0;
}
