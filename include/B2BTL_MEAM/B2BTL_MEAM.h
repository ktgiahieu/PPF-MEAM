#ifndef B2BTL_MEAM
#define B2BTL_MEAM

#include "dataType.h"

#include <pcl/common/common.h>
// kdtree
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/surface/mls.h>
//extract indices
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
//opencv
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
//Eigen
#include <Eigen/Dense>


using namespace cv;

void cloudEdgeDetection(const PointCloudType::ConstPtr& cloud_source, const PointCloudType::ConstPtr& cloud_extracted, Mat image, PointCloudType::Ptr& cloud_edge);

void tangentLine(const PointCloudType::ConstPtr& cloud_in, pcl::PointCloud<PointXYZTangent>::Ptr& cloud_out);

#endif