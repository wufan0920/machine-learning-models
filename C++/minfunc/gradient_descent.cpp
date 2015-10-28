#include <iostream>
#include "include/minfunc.h"

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

void MinfuncDescent::optimize(MatrixXd& data, MatrixXd& label, MatrixXd& weight, Gradient& gradfunc)
{
    float n = data.rows();
    float alpha = this->alpha;
    int iters = this->iters;
    int rows = weight.rows();
    int cols = weight.cols();
    MatrixXd grad(rows,cols);

    for(int index = 0; index < iters; index++)
    {
        cout<<"iters:"<<index<<endl;
        grad = gradfunc.compute_grad(data, label, weight);
        weight = weight - alpha * grad;
    }

    return;
}
