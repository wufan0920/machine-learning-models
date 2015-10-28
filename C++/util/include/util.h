#include <Dense>
#include <string>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

vector<string>& split(string& s, string& delim, vector<string>& ret);
void load_data(MatrixXd& data, ifstream& stream);
void add_bias(MatrixXd* pDest,MatrixXd* pSrc);
MatrixXd operator/(MatrixXd origin, VectorXd divide);
MatrixXd colwise_minus(MatrixXd origin, VectorXd minus);
MatrixXd rowwise_add(MatrixXd origin, VectorXd add);
VectorXd stDeviation(MatrixXd data, int axis);
double stDeviation(MatrixXd data);
MatrixXd sigmoid(MatrixXd& input);
MatrixXd sigInv(MatrixXd& input);
double vector_innerproduct(MatrixXd v, MatrixXd p);
