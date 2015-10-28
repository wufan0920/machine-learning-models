#include <iostream>
#include <Dense>
#include <util.h>
#include "include/minfunc.h"

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

void MinfuncNewton::optimize(MatrixXd& data, MatrixXd& label, MatrixXd& weight, Gradient& gradfunc)
{
    int n = data.rows();
    int iters = this->iters;
    MatrixXd score, h, H, grad;
    MatrixXd diag, diag2;

    for(int index = 0; index < iters; index++)
    {
        score = data * weight;
        h = sigmoid(score);
        diag = h.asDiagonal();
        MatrixXd tmp = 1 - h.array();
        diag2 = tmp.asDiagonal();
        grad = gradfunc.compute_grad(data, label, weight);
        H = (1.0/n) * data.transpose() * diag * diag2 * data;
        weight = weight - (H.inverse() * grad);
    }
    return;
}
