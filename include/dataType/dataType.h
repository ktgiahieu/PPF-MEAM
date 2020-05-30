#ifndef DATATYPE
#define DATATYPE

#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/ppf.h>

# define PI  3.1415926
using namespace std;
const int descr_type = 2;// 0 SHOT, 1 PPF, 2 B2BTL_MEAM

typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointCloud<PointType> PointCloudType;
typedef pcl::Normal NormalType;
typedef pcl::PointCloud<NormalType> PointCloudNormalType;
typedef pcl::ReferenceFrame RFType;

typedef pcl::SHOT352 DescriptorTypeSHOT;
typedef pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorTypeSHOT> EstimatorTypeSHOT;

typedef pcl::PPFSignature DescriptorTypePPF;
typedef pcl::PPFEstimation<PointType, NormalType, DescriptorTypePPF> EstimatorTypePPF;

struct PointXYZTangent
{
	PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
	float tx;
	float ty;
	float tz;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZTangent,           // here we assume a XYZ + "test" (as fields)
(float, x, x)
(float, y, y)
(float, z, z)
(float, tx, tx)
(float, ty, ty)
(float, tz, tz)
)

struct CloudStyle
{
	double r;
	double g;
	double b;
	double size;

	CloudStyle(double r,
		double g,
		double b,
		double size) :
		r(r),
		g(g),
		b(b),
		size(size)
	{
	}
};

extern CloudStyle style_white;
extern CloudStyle style_red;
extern CloudStyle style_green;
extern CloudStyle style_cyan;
extern CloudStyle style_violet;

#endif