#ifndef KINECT
#define KINECT

#include "pclFunction.h"
#include "meshSampling.h"
#include "HPR.h"
#include "B2BTL_MEAM.h"

using namespace pcl;
using namespace cv;

class Kinect
{
private:
	//Intrinsics
	double rgb_fx = 527.360558216633;
	double rgb_fy = 528.3632079059613;
	double rgb_cx = 319.6283688314646;
	double rgb_cy = 263.2500840924527;

	double depth_fx = 585.3081185218954;
	double depth_fy = 587.28826280342;
	double depth_cx = 317.8356267919822;
	double depth_cy = 247.1889948214958;

	//IO
	std::string model_filename_ = "../../../data/nap.STL";
	bool is_mesh = true;

	//Algorithm params
	bool show_keypoints_ = true;
	bool show_corr = false;
	bool show_FP = false;
	bool show_instances = false;
	bool use_cloud_resolution_ = false;
	bool use_hough_ = true;
	float samp_rad = 0.01f;
	float norm_rad = 2.5 * samp_rad;
	float rf_rad_ = 0.75f * norm_rad;
	float descr_rad_ = 1.0f * norm_rad;
	float cg_size_ = 0.05f;
	float cg_thresh_ = 5.0f;

	float sqr_descr_dist = 0.25f;
	int icp_max_iter_ = 5;
	float icp_corr_distance_ = 0.005f;
	float hv_resolution_ = 0.005f;
	float hv_occupancy_grid_resolution_ = 0.01f;
	float hv_clutter_reg_ = 5.0f;
	float hv_inlier_th_ = 0.05f;
	float hv_occlusion_th_ = 0.1f;
	float hv_rad_clutter_ = 0.03f;
	float hv_regularizer_ = 3.0f;
	float hv_rad_normals_ = 0.05;
	bool hv_detect_clutter_ = true;
	
	//Grabber
	pcl::io::OpenNI2Grabber* interface = new pcl::io::OpenNI2Grabber("B00364608306123B");
	std::queue<Mat> images_q;
	const int max_images = 10;

	// Model 
	PointCloudType::Ptr model = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr model_keypoints = PointCloudType::Ptr(new PointCloudType());

	pcl::PointCloud<NormalType>::Ptr model_normals = pcl::PointCloud<NormalType>::Ptr(new pcl::PointCloud<NormalType>());
	pcl::PointCloud <pcl::PointNormal>::Ptr model_keypoints_with_normals = pcl::PointCloud <pcl::PointNormal>::Ptr(new pcl::PointCloud <pcl::PointNormal>());
	pcl::PointCloud<DescriptorTypePPF>::Ptr model_descriptors_PPF = pcl::PointCloud<DescriptorTypePPF>::Ptr(new pcl::PointCloud<DescriptorTypePPF>());
	pcl::PointCloud<DescriptorTypeSHOT>::Ptr model_descriptors_SHOT = pcl::PointCloud<DescriptorTypeSHOT>::Ptr(new pcl::PointCloud<DescriptorTypeSHOT>());
	pcl::PointCloud<RFType>::Ptr model_rf = pcl::PointCloud<RFType>::Ptr(new pcl::PointCloud<RFType>());
	pcl::PPFHashMapSearch::Ptr ppf_hashmap_search = pcl::PPFHashMapSearch::Ptr(new PPFHashMapSearch(12.0f / 180.0f * float(M_PI), 0.05f));
	pcl::PointCloud<PointType>::Ptr model_good_kp = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

	pcl::PointCloud<PointType>::Ptr off_scene_model = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
	
	//Scene
	PointCloudType::Ptr captured_scene = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr passthroughed_scene = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr voxelgrid_filtered_scene = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr segmented_scene = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr statistical_filtered_scene = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr scene = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr scene_keypoints = PointCloudType::Ptr(new PointCloudType());
	pcl::PointCloud<NormalType>::Ptr scene_normals = pcl::PointCloud<NormalType>::Ptr(new pcl::PointCloud<NormalType>());
	
	pcl::PointCloud<RFType>::Ptr scene_rf = pcl::PointCloud<RFType>::Ptr(new pcl::PointCloud<RFType>());
	pcl::PointCloud<PointType>::Ptr scene_good_kp = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr off_model_good_kp = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());


	//Others
	int prev_instances_size = 0; // for removing previous cloud
	boost::mutex mtx; //Mutex for locking
	pcl::console::TicToc tt; // Tictoc for process-time calculation

public:
	CustomVisualizer customViewer;
	Kinect(){}
	void initIntrinsics();
	bool run();
	void stop();
	bool isRunning();
	bool loadModel();
	bool prepareModelSHOT();
	bool prepareModelPPF();
	bool prepareModelB2BTL_MEAM();
	void _3D_Matching_SHOT(const PointCloudType::ConstPtr &cloud);
	void _3D_Matching_PPF(const PointCloudType::ConstPtr &cloud);
	void _3D_Matching_B2BTL_MEAM(const PointCloudType::ConstPtr &cloud);
};

#endif