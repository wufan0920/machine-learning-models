#ifndef GRADIENT_H
#define GRADIENT_H

#include <Dense>
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

class Gradient
{
    public:
        Gradient(){}
        ~Gradient(){}
        virtual MatrixXd compute_grad(MatrixXd& data, MatrixXd& label, MatrixXd& weight) = 0;
};

class RegressionGradient : public Gradient
{
    public:
        RegressionGradient(){}
        ~RegressionGradient(){}
        MatrixXd compute_grad(MatrixXd& data, MatrixXd& label, MatrixXd& weight);
};

class NewtonGradient : public Gradient
{
    public:
        NewtonGradient(){}
        ~NewtonGradient(){}
        MatrixXd compute_grad(MatrixXd& data, MatrixXd& label, MatrixXd& weight);
};

class NeuralNetworkGradient : public Gradient
{
    public:
        NeuralNetworkGradient():lambda(0),sparsity_param(0),beta(0),visible_size(0),hidden_size(0){}
        NeuralNetworkGradient(float lmd, float sparsity, float beta, int visible_size, int hidden_size)
        {
            this->lambda = lmd;
            this->sparsity_param = sparsity;
            this->beta = beta;
            this->visible_size = visible_size;
            this->hidden_size = hidden_size;
        }
        ~NeuralNetworkGradient(){}
        MatrixXd compute_grad(MatrixXd& data, MatrixXd& label, MatrixXd& weight);
    private:
        float lambda;
        float sparsity_param;
        float beta;
        int visible_size;
        int hidden_size;
};

#endif 
