#include "common.h"
#include <iostream>
//#include "Target.h"
//just  for  test
// compute the log of the Gaussian probability
float log_gaussian_prob(float x, float m, float std)
{
	if(std == 0)
		return 0.0;
	else
		return -log(sqrt(2 * M_PI) * std) - ( pow((x - m) / std, 2) / 2.0 );
}

void drawDashRect(cv::Mat &img, int lineLen, int dashLen, cv::Rect rect, cv::Scalar& color, int thickness) {
	int totalLen = lineLen + dashLen;
	int nX = rect.width / totalLen;
	int nY = rect.height / totalLen;

	cv::Point start = rect.tl(), end = rect.tl();

	for (int i = 0; i < nX; i++) {
		//draw top line
		start = cv::Point(rect.x + i*totalLen, rect.y);
		end = cv::Point(rect.x + (i + 1)*totalLen - dashLen, rect.y);
		cv::line(img, start, end, color, thickness);
		//draw bottom line
		start.y = rect.y + rect.height;
		end.y = start.y;
		cv::line(img, start, end, color, thickness);
	}
	for (int i = 0; i < nY; i++) {
		//draw top line
		start = cv::Point(rect.x, rect.y + i*totalLen);
		end = cv::Point(rect.x, rect.y + (i + 1)*totalLen - dashLen);
		cv::line(img, start, end, color, thickness);
		//draw bottom line
		start.x = rect.x + rect.width;
		end.x = start.x;
		cv::line(img, start, end, color, thickness);
	}
}

bool getImgPatchBySize(const cv::Mat & img, cv::Size sz, cv::Rect & orgRect, cv::Mat & patch){
	//fix me @ 2017/06/14  foolslove
	cv::Rect roi = orgRect;
	if (roi.x - 4 <= 0 || roi.y - 4 <= 0 || roi.x + 4 >= img.cols || roi.y + 4 >= img.rows) 
		return false;

	cv::Size _sz = sz;
	cv::Rect rect = roi;
	rect.x = roi.x - roi.width / 2;
	rect.y = roi.y - roi.height/ 2;

	//小于指定大小时，直接扩展矩形框
	if (roi.width <= sz.width && roi.height <= sz.height)
		rect = enlargeRect(rect, sz);
	else
		_sz = cv::Size(roi.width, roi.height);  //否则按实际大小截取矩形区域
	roi = rect & cv::Rect(0, 0, img.cols, img.rows);
	if (roi == cv::Rect(0, 0, 0, 0))
		return false;
	
	cv::Mat tmp = cv::Mat::zeros(_sz, CV_8UC1);
	int cx = rect.x < 0 ? -rect.x : 0;
	int cy = rect.y < 0 ? -rect.y : 0;
	cv::Rect r(cx, cy, roi.width, roi.height);
	img(roi).copyTo(tmp(r));		
	if (tmp.rows > sz.height || tmp.cols > sz.width)
		resize(tmp, patch, sz);
	else
		tmp.copyTo(patch);

	return true;
}

cv::Rect enlargeRect(cv::Rect rect, cv::Size &sz) {
	cv::Point center(rect.x + rect.width/2,rect.y+rect.height/2);
	int maxLen = std::max(rect.width, rect.height);
	if (maxLen > sz.width) {
		sz.width = maxLen;
		sz.height = maxLen;
	}

	return cv::Rect(center.x - sz.width / 2, center.y - sz.height / 2, sz.width, sz.height);
}

void readDict(std::string& path, Matrix_t& PosDict, Matrix_t& NegDict, Matrix_t& PosG, Matrix_t& NegG) {
	cv::Mat dict1, dict2;
	cv::FileStorage fs(path, cv::FileStorage::READ);

	if (!fs.isOpened()) {
		std::cout << "failed to open file " << path << std::endl;
		return ;
	}

	fs["posDict"] >> dict1;
	fs["negDict"] >> dict2;
	fs.release();

	PosDict.resize(dict1.rows, dict1.cols);
	if (dict1.isContinuous()) {
		for (int i = 0; i < dict1.rows; i++) {
			for (int j = 0; j < dict1.cols; j++) {
				PosDict(i, j) = dict1.data[i*dict1.rows + j];
			}
		}
	}
	PosG = PosDict.transpose()*PosDict;

	NegDict.resize(dict2.rows, dict2.cols);
	if (dict2.isContinuous()) {
		for (int i = 0; i < dict2.rows; i++) {
			for (int j = 0; j < dict2.cols; j++) {
				NegDict(i, j) = dict2.data[i*dict2.rows + j];
			}
		}
	}
	NegG = NegDict.transpose()*NegDict;
}

void OMPSteps(const Matrix_t & dict, Vector_t & input, int sparcity, float& err)
{
	const Scalar_t Epsilon = (Scalar_t)1e-4;
	int dimensionality = dict.rows();
	int dictionary_size = dict.cols();

	Vector_t r = input;							// residual
	Vector_t recon;								// recovery patch
	IntArray_t I_atoms;								// (out) list of selected atoms for given sample
	Matrix_t L(1, 1);								// Matrix from Cholesky decomposition, incrementally augmented
	L(0, 0) = (Scalar_t)1.;
	int I_atom_count = 0;

	Matrix_t dk(dimensionality, 1);
	Matrix_t DictI_T(0, dimensionality);			// Incrementaly updated
	Matrix_t xI;	
	bool jumpOut = false;
	// (out) -> encoded signal
	for (int k = 0; k < sparcity; k++)
	{
		// Project residual on all dictionary atoms (columns), find the one that match best
		int max_idx = -1;
		Scalar_t max_value = (Scalar_t)-1.;

#pragma omp parallel for shared(max_value,max_idx), private(jumpOut)
		for (int atom_idx = 0; atom_idx < dictionary_size; atom_idx++)
		{
			jumpOut = false;
			for (int i = 0; i < I_atoms.size(); i++)
				if (atom_idx == I_atoms(i)) {
					jumpOut = true;
					break;
				}

			if(jumpOut) continue;
			//std::cout << "Here is the atom " << atom_idx << " :" << dict.col( atom_idx ) << std::endl;
			//std::cout << "r:" << r << std::endl;
			Scalar_t dot_val = fabs((Scalar_t)dict.col(atom_idx).dot(r));
#pragma omp critical
			{
				if (dot_val > max_value)
				{
					max_value = dot_val;
					max_idx = atom_idx;
				}
			}
		}
		if (max_value < Epsilon)
			break;

		// We need to solve xI = DictI+.input
		// where pseudo inverse DictI+ = (DictI_T.DictI)^(-1).DictI_T
		// so xI = (DictI_T.DictI)^(-1).alpha_I where alpha_I = DictI_T.input

		if (I_atom_count >= 1)
		{
			dk.col(0) = dict.col(max_idx);
			Matrix_t DITdk = DictI_T * dk;

			// w = solve for w { L.w = DictIT.dk }
			Matrix_t w = L.triangularView<Eigen::Lower>().solve(DITdk);

			//            | L       0		|
			// Update L = | wT  sqrt(1-wTw)	|
			//                               
			L.conservativeResize(I_atom_count + 1, I_atom_count + 1);
			L.row(I_atom_count).head(I_atom_count) = w.col(0).head(I_atom_count);
			L.col(I_atom_count).setZero();

			Scalar_t val_tmp = 1 - w.col(0).dot(w.col(0));
			if (val_tmp <= 1e-8)
				break;
			L(I_atom_count, I_atom_count) = val_tmp < 1 ? (Scalar_t) ::sqrt((Scalar_t)val_tmp) : 1;
		}

		I_atoms.conservativeResize(I_atom_count + 1);
		I_atoms[I_atom_count] = max_idx;

		DictI_T.conservativeResize(I_atom_count + 1, dimensionality);
		DictI_T.row(I_atom_count) = dict.col(max_idx);
		I_atom_count++;

		Matrix_t alpha_I(I_atom_count, 1);
		alpha_I = DictI_T * input;

		// xI = solve for c { L.LT.c = alpha_I }
		// first solve LTc :
		Matrix_t LTc = L.triangularView<Eigen::Lower>().solve(alpha_I);
		//std::cout << "L: " << L<<"\nalpha_I: " << alpha_I << "\nLTc: " << LTc << std::endl;
		// then solve xI :
		xI = L.transpose().triangularView<Eigen::Upper>().solve(LTc);
		//std::cout << "sampleId: "<< sample_idx<<"sparseId: " << k <<"\n"<<xI << std::endl;
		// r = y - Dict_I * xI
		r = input - DictI_T.transpose() * xI;
		//std::cout << xI << r << std::endl;
	}

	recon = Vector_t::Zero(input.rows(), 1);
	bool isNoise = true;
	for (int i = 0; i < I_atoms.size(); i++) {
		std::cout << "alph: :" << xI(i);
		if (I_atoms(i) < sparcity) {
			recon += xI(i)*DictI_T.row(i);
			isNoise = false;
		}
	}
	std::cout << std::endl;
	if (isNoise)
		err = 10e6;
	else {
		r = input - recon;
		err = r.squaredNorm();
	}
}

void orthonormalize(Matrix_t & ColVecs)  
{  
    ColVecs.col(0).normalize();  
    double temp;  
    for(std::size_t k = 0; k != ColVecs.cols() - 1; ++k)  
    {  
        for(std::size_t j = 0; j != k + 1; ++j)  
        {  
            temp = ColVecs.col(j).transpose() * ColVecs.col(k + 1);  
            ColVecs.col(k + 1) -= ColVecs.col(j) * temp;  
        }  
        ColVecs.col(k + 1).normalize();  
    }  
}

void SKLStep(Matrix_t & U, Vector_t & E, Matrix_t & B, Vector_t& mu, int& colNum, int spanNum, float ff){
	Matrix_t R, noZeroB;
	Matrix_t B_tmp,B_Proj;
	Matrix_t U1;
	Vector_t E1;
	Eigen::JacobiSVD<Matrix_t> mySvd;

	Vector_t mu0 = mu;
	int newCols = B.cols();

	mu = B.rowwise().mean(); 
	if (colNum > 0) {
		noZeroB.resize(B.rows(), newCols + 1);
		noZeroB.leftCols(newCols) = B - mu.replicate(1, newCols);
		noZeroB.col(newCols) = std::sqrt(colNum*newCols*1.0 / (colNum + newCols))*(mu - mu0);
		mu = (ff*colNum*mu0 + newCols*mu) / (newCols + ff*colNum);
		colNum = newCols + ff*colNum;

		B_Proj = U.transpose()*noZeroB;
		B_tmp = noZeroB - U*B_Proj;

		Eigen::HouseholderQR<Matrix_t> qr;
		qr.compute(B_tmp);
		Matrix_t q = qr.householderQ();
		Matrix_t Q;
		Matrix_t q_res;

		q_res = q.leftCols(B_tmp.cols());
		Q.resize(U.rows(), U.cols() + B_tmp.cols());
		Q.leftCols(U.cols()) = U;
		Q.rightCols(B_tmp.cols()) = q_res;

		R.setZero(E.rows() + B_Proj.cols(), E.rows() + B_Proj.cols());
		R.topLeftCorner(E.rows(), E.rows()) = ff*E.asDiagonal();
		R.topRightCorner(B_Proj.rows(), B_Proj.cols()) = B_Proj;
		R.bottomRightCorner(B_tmp.cols(), B_tmp.cols()) = q_res.transpose()*B_tmp;
		mySvd.compute(R, Eigen::ComputeThinU | Eigen::ComputeThinV);

		U1 = mySvd.matrixU();
		E1 = mySvd.singularValues();
		spanNum = std::min(spanNum, (int)E1.size());
		U = Q*U1.leftCols(spanNum);
		E = E1.head(spanNum);
	}
	else {
		colNum += B.cols();
		B = B - mu.replicate(1, newCols);
		mySvd.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
		U1 = mySvd.matrixU();
		E1 = mySvd.singularValues();

		spanNum = std::min(spanNum, (int)E1.size());
		U = U1.leftCols(spanNum);
		E = E1.head(spanNum); 
	}	
}

void PCA_L1(Vector_t & data, Matrix_t & U, float lamda, float & err){
#define  MAX_ITERATE 50
#define  TOLERANCE 1e-3
	Vector_t coeffZ = Vector_t::Zero(U.cols(), 1);
	Vector_t coeffE = Vector_t::Zero(U.rows(), 1);
	Vector_t tmpErr = Vector_t::Zero(data.rows(), 1);
	float sumErr = 0., objValue = 0., objValueCur = 0.;

	if (U.cols() == MAX_BASIS) {
		for (int i = 0; i < MAX_ITERATE; i++) {
			sumErr = 0.;
			coeffZ = U.transpose()*(data - coeffE);
			tmpErr = data - U * coeffZ;
			for (int j = 0; j < tmpErr.rows(); j++) {
				float tmp = std::max(abs(tmpErr(j)) -(float) ERROR_THRESH, (float)0.);
				sumErr += tmp;
				coeffE(j) = tmpErr(j) >= 0 ? tmp : -tmp;
			}

			if (i > 0) {
				tmpErr = tmpErr - coeffE;
				objValueCur = tmpErr.transpose()*tmpErr + lamda*sumErr;
				if (abs(objValueCur - objValue) < TOLERANCE)
					break;
				else
					objValue = objValueCur;
			}			
		}

		sumErr = 0;
		for (int i = 0; i < tmpErr.rows(); i++) {
			if (abs(coeffE(i)) >= ERROR_THRESH) {
				tmpErr(i) = 0;
				sumErr++;
			}
		}
		sumErr *= lamda;
		sumErr += tmpErr.squaredNorm();
	}
	else {
		coeffZ = U.transpose()*data;
		tmpErr = data - U*coeffZ;
		sumErr = tmpErr.squaredNorm();
	}
	//fout << coeffZ.transpose() << "coffe: " << coeffE.transpose();
	err = sumErr;
	data = coeffE;
}

float testOne(cv::Mat& img, cv::Ptr<cv::ml::SVM> svm, cv::HOGDescriptor& hog){
	std::vector<float> descriptor;

	hog.compute(img, descriptor, cv::Size(7, 7));
	int DescriptorDim = descriptor.size();
	//cout << "特征维数为 : " << DescriptorDim << endl;
	cv::Mat TestHog = cv::Mat::zeros(1, descriptor.size(), CV_32FC1);
	for (int i = 0; i < DescriptorDim; i++)
	{
		TestHog.at<float>(0, i) = descriptor[i];
	}
	return svm->predict(TestHog);
}

float entropy(std::vector<int>& bins, int num)
{
	float entropyV = 0.0;
	for (int i = 0; i < bins.size(); i++) {
		if(bins[i] <= 0) continue;
		float prob = bins[i] * 1.0 / num;
		entropyV -= prob*log10(prob);
	}
	return entropyV;
}
