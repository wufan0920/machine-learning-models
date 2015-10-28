#include <iostream>
#include <vector>
#include <Dense>
#include <util.h>
#include "include/minfunc.h"

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

MatrixXd lbfgs_descent(MatrixXd grad, vector<MatrixXd>& old_directions, vector<MatrixXd>& old_steps, double hession)
{
    MatrixXd descent;
    int keep = old_steps.size();
    int size = old_steps.front().size();

    MatrixXd ro = MatrixXd::Zero(keep, 1);
    MatrixXd q = MatrixXd::Zero(size, keep + 1);
    MatrixXd r = MatrixXd::Zero(size, keep + 1);
    MatrixXd al = MatrixXd::Zero(keep, 1);
    MatrixXd be = MatrixXd::Zero(keep, 1);

    for(int index = 0; index < keep; index++)
    {
        ro(index) = 1/vector_innerproduct(old_steps[index], old_directions[index]);
    }

    q.rightCols(1) = grad;
    
    for(int index = keep - 1; index >= 0; index--)
    {
        al(index) = ro(index)*vector_innerproduct(old_directions[index], q.col(index+1)); 
        q.col(index) = q.col(index + 1) - al(index)*old_steps[index];
    }

    r.col(0) = hession*q.col(0);

    for(int index = 0; index < keep; index++)
    {
        be(index) = ro(index)*vector_innerproduct(old_steps[index], r.col(index));
        r.col(index+1) = r.col(index) + old_directions[index]*(al(index) - be(index));
    }

    descent = r.rightCols(1);
    return descent;
}

void lbfgs_update(MatrixXd y, MatrixXd s, vector<MatrixXd>& old_directions, vector<MatrixXd>& old_steps, double keep, double& hession)
{
    double score = vector_innerproduct(y,s);

    if(score > 1e-10)
    {
        int size = old_steps.size();
        if(size >= keep)
        {
            old_steps.erase(old_steps.begin());
            old_directions.erase(old_directions.begin());
        }

		old_steps.push_back(y);
		old_directions.push_back(s);

        hession = score/vector_innerproduct(y,y);
    }
    else
    {
        cout<<"NG"<<endl;
    }
    return;
}

void MinfuncLBFGS::optimize(MatrixXd& data, MatrixXd& label, MatrixXd& weight, Gradient& gradfunc)
{
	int iters = this->iters;
	float alpha = this->alpha;
	int keep = this->keep; 

    MatrixXd grad, descent, old_grad; 
    vector<MatrixXd> old_directions, old_steps;
    double hession;

    for(int index = 0; index < iters; index++)
    {
        cout<<"turn: "<<index<<endl;
		grad = gradfunc.compute_grad(data, label, weight);
        if(index == 0)
        {
            descent = -grad;
            hession = 1;
        }
        else
        {
            lbfgs_update(grad - old_grad, alpha*descent, old_directions, old_steps, keep, hession);
            descent = lbfgs_descent(-grad, old_directions, old_steps, hession);
        }

        old_grad = grad;
        weight = weight + alpha*descent;
    } 
}
