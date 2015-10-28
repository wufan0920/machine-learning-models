#include <iostream>
#include <fstream>
#include <Dense>
#include <string>
#include <vector>
#include <math.h>
#include "stdlib.h"
#include "include/util.h"

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

vector<string>& split(string& s, string& delim, vector<string>& ret)
{  
    size_t head = s.find_first_not_of(delim,0);  
    while (head != string::npos)  
    {  
        size_t tail = s.find_first_of(delim, head);
        ret.push_back(s.substr(head, tail));  
        head = s.find_first_not_of(delim, tail);  
    }  
      
    return ret;
}

void load_data(MatrixXd& data, ifstream& stream)
{
    char buffer[1024*1024];
    int row = 0;
    while(NULL != stream.getline(buffer,1024*1024))
    {
        string delim(" ");
        string line(buffer);
        vector<string> elements;
        elements = split(line, delim, elements);
        int size = elements.size();

        for(int col = 0; col < size; col++)
        {
            float tmp = atof(elements.at(col).c_str());
            data(row,col) = tmp;
        }
        row++;
    }
    
    return;
}

void add_bias(MatrixXd* pDest,MatrixXd* pSrc)
{
    int rows = pDest->rows();
    int cols = pDest->cols();
 
    MatrixXd bias(rows,1);
	bias = MatrixXd::Ones(rows,1);
 
    pDest->block(0,0,rows,cols-1) = *pSrc;
    pDest->col(cols-1) = bias;
    return;
}

MatrixXd operator/(MatrixXd origin, VectorXd divide)
{
    int col = origin.cols();
    int row = origin.rows();

    for(int index = 0; index < col; index++)
    {
        origin.col(index) /= divide(index);
    }
    
    return origin;
}

MatrixXd colwise_minus(MatrixXd origin, VectorXd minus)
{
    int col = origin.cols();
    int row = origin.rows();
    MatrixXd tominus(row,col);
    for(int index = 0; index < row; index++)
    {
        tominus.row(index) = minus;
    }

    origin -= tominus;
    
    return origin;
}

MatrixXd rowwise_add(MatrixXd origin, VectorXd add)
{
    int col = origin.cols();
    int row = origin.rows();
    MatrixXd toadd(row,col);
    for(int index = 0; index < col; index++)
    {
        toadd.col(index) = add;
    }

    origin += toadd;
    
    return origin;
}

double stDeviation(MatrixXd data)
{
    double sum;
    double deviation;
    double mean = data.mean();
    int num = data.cols() * data.rows();
    MatrixXd tmp = data.array() - mean;
    tmp = tmp.array().square();
    sum = tmp.sum()/num;
    deviation = sqrt(sum);
    
    return deviation;
}

VectorXd stDeviation(MatrixXd data, int axis)
{
    //TODO:0.col wise std; 1.row wise std;
    int col = data.cols();
    int row = data.rows();
    VectorXd means = data.colwise().mean();
    VectorXd deviation(col);

    data = colwise_minus(data,means);

    //square the data
    data = data.array().square();
    
    deviation = data.colwise().sum();
    deviation /= row;
    
    //sqrt
    deviation = deviation.array().sqrt();

    return deviation;
}

MatrixXd sigmoid(MatrixXd& input)
{
    return ((-input).array().exp().array()+1).array().inverse();
}

MatrixXd sigInv(MatrixXd& input)
{
    MatrixXd tmp = ((-sigmoid(input)).array() + 1);
    return sigmoid(input).cwiseProduct(tmp);
}

double vector_innerproduct(MatrixXd v, MatrixXd p)
{
   int size = v.size();
   v.resize(1,size);
   p.resize(size,1);
   return (v*p)(0);
}
