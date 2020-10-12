#ifndef B2BTL_MEAM
#define B2BTL_MEAM

#include "pclFunction.h"
#include "meshSampling.h"
#include "HPR.h"
#include "B2BTL_MEAMRegistration.h"

#include <pcl/common/common.h>
// kdtree
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/surface/mls.h>
//extract indices
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
//
#include <pcl/features/boundary.h>
//opencv
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/aruco.hpp"

//Eigen
#include <Eigen/Dense>

#include <mutex>

class DescriptorB2BTL_MEAM
{
private:
	//IO
	std::string model_filename_;
	std::string type = "B2BTL_MEAM";
	bool show_FalsePose = false;

	//Algorithm params
	double t_sampling = 0.02;
	float samp_rad;
	float norm_rad;
	float Lvoxel_encode;
	double Dthresh = 0.0008; //0.8mm
	float Sthresh = 0.18;
	float angle_discretization_step = 12.0f / 180.0f * float(M_PI);
	float distance_discretization_step = 0.005f;
	float hv_dist_thresh = distance_discretization_step;
	int scene_reference_point_sampling_rate = 1;
	int icp_max_iter_ = 20;
	float icp_corr_distance_ = 0.01f;
	size_t Npv = 4;

	// Model 
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_sampling = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr model = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	std::vector< pcl::PointCloud<PointXYZTangent>::Ptr> MEAM;
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<PointXYZTangent>::Ptr model_keypoints_tangent = pcl::PointCloud<PointXYZTangent>::Ptr(new pcl::PointCloud<PointXYZTangent>());
	pcl::B2BTL_MEAMHashMapSearch::Ptr b2btl_meam_hashmap_search = pcl::B2BTL_MEAMHashMapSearch::Ptr(new pcl::B2BTL_MEAMHashMapSearch(angle_discretization_step, distance_discretization_step));

	//Others
	pcl::console::TicToc tt; // Tictoc for process-time calculation
	std::mutex mtx;
	cv::Mat latestImage;
	PointCloudType::Ptr latestCloud = PointCloudType::Ptr(new PointCloudType());
	
public:
	CustomVisualizer customViewer;

	DescriptorB2BTL_MEAM() {}
	std::string getType();
	void setModelPath(std::string model_path_);
	void cloudEdgeDetection(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_source, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_extracted, cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_edge);
	void tangentLine(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, pcl::PointCloud<PointXYZTangent>::Ptr& cloud_out);
	bool loadModel();
	bool prepareModelDescriptor();
	void storeLatestCloud(const PointCloudType::ConstPtr &cloud);
	void storeLatestImage(cv::Mat& image);
	void _3D_Matching();
};

#endif