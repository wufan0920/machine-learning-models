#include <Dense>
#include "gradient.h"
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

class Minfunc
{
    public:
        Minfunc(){}
        ~Minfunc(){}
        virtual void optimize(MatrixXd& data, MatrixXd& label, MatrixXd& weight, Gradient& gradfunc) = 0;
    protected:
        int iters;
};

class MinfuncDescent : public Minfunc
{
    public:
        MinfuncDescent(){this->iters = 100; this->alpha = 0.01;}
        MinfuncDescent(int iters, float alpha)
        {
            this->iters = iters;
            this->alpha = alpha;
        }

        ~MinfuncDescent(){}
        void optimize(MatrixXd& data, MatrixXd& label, MatrixXd& weight, Gradient& gradfunc);
    private:
        float alpha;
};

class MinfuncNewton : public Minfunc
{
    public:
        MinfuncNewton(){this->iters = 100;}
        MinfuncNewton(int iters){ this->iters = iters;}
        ~MinfuncNewton(){}
        void optimize(MatrixXd& data, MatrixXd& label, MatrixXd& weight, Gradient& gradfunc);
};

class MinfuncLBFGS : public Minfunc
{
    public:
        MinfuncLBFGS()
        {
            this->iters = 100;
            this->alpha = 0.01;
            this->keep = 100;
        }

        MinfuncLBFGS(int iters, float alpha, int keep)
        {
            this->iters = iters;
            this->alpha = alpha;
            this->keep = keep;
        }
        ~MinfuncLBFGS(){}
        void optimize(MatrixXd& data, MatrixXd& label, MatrixXd& weight, Gradient& gradfunc);
    private:
        float alpha;
        int keep;
};

