#ifndef PPF
#define PPF

#include "pclFunction.h"
#include "meshSampling.h"
#include "HPR.h"
#include "MyPPFRegistration.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

class DescriptorPPF
{
private:
	//IO
	std::string model_filename_ = "D:/BK/HK192/LVTN/Code/BinPicking/data/6914.STL";
	bool is_mesh = true;
	std::mutex mtx;
	bool use_kinect = false;

	cv::Mat latestImage;
	PointCloudType::Ptr latestCloud = PointCloudType::Ptr(new PointCloudType());

	//Algorithm params
	double t_sampling = 0.04;
	float samp_rad;
	float norm_rad;
	double Lvoxel;
	bool show_FP = true;

	std::vector<float> passthrough_limits = { -0.5, 0.5, -0.5, 0.5, -0.1, 1.0 };
	float angle_discretization_step = 12.0f / 180.0f * float(M_PI);
	float distance_discretization_step = 0.005f;
	int scene_reference_point_sampling_rate = 10;
	int scene_referred_point_sampling_rate = 5;
	float hv_dist_thresh = distance_discretization_step;
	size_t Npv = 4;

	int icp_max_iter_ = 1;
	float icp_corr_distance_ = 0.01f;
	float hv_resolution_ = 0.005f;
	float hv_occupancy_grid_resolution_ = 0.01f;
	float hv_clutter_reg_ = 5.0f;
	float hv_inlier_th_ = 0.05f;
	float hv_occlusion_th_ = 0.1f;
	float hv_rad_clutter_ = 0.03f;
	float hv_regularizer_ = 3.0f;
	float hv_rad_normals_ = 0.05;
	bool hv_detect_clutter_ = true;

	std::string type = "PPF";

	// Model 
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_sampling = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr model = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud <pcl::PointNormal>::Ptr model_keypoints_with_normals = pcl::PointCloud <pcl::PointNormal>::Ptr(new pcl::PointCloud <pcl::PointNormal>());

	pcl::PointCloud<PointXYZTangent>::Ptr off_scene_model_keypoints_with_normals = pcl::PointCloud<PointXYZTangent>::Ptr(new pcl::PointCloud<PointXYZTangent>());

	pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model_keypoints = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	std::shared_ptr<pcl::MyPPFHashMapSearch> ppf_hashmap_search = std::make_shared<pcl::MyPPFHashMapSearch>(angle_discretization_step, distance_discretization_step);

	//Others
	int prev_instances_size = 0; // for removing previous cloud
	pcl::console::TicToc tt; // Tictoc for process-time calculation

	
public:
	CustomVisualizer customViewer;
	DescriptorPPF(bool use_kinect_in):use_kinect(use_kinect_in) {}

	std::string getType();
	bool loadModel();
	bool prepareModelDescriptor();
	void storeLatestCloud(const PointCloudType::ConstPtr &cloud);
	void storeLatestImage(cv::Mat& image);
	void _3D_Matching();
};



#endif