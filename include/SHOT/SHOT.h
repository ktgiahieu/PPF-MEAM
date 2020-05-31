#ifndef SHOT
#define SHOT

#include "pclFunction.h"
#include "meshSampling.h"

class DescriptorSHOT
{
private:
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
	int icp_max_iter_ = 1;
	float icp_corr_distance_ = 0.05f;
	float hv_resolution_ = 0.005f;
	float hv_occupancy_grid_resolution_ = 0.01f;
	float hv_clutter_reg_ = 5.0f;
	float hv_inlier_th_ = 0.05f;
	float hv_occlusion_th_ = 0.1f;
	float hv_rad_clutter_ = 0.03f;
	float hv_regularizer_ = 3.0f;
	float hv_rad_normals_ = 0.05;
	bool hv_detect_clutter_ = true;

	std::string type = "SHOT";

	// Model 
	PointCloudType::Ptr model = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr model_keypoints = PointCloudType::Ptr(new PointCloudType());

	pcl::PointCloud<NormalType>::Ptr model_normals = pcl::PointCloud<NormalType>::Ptr(new pcl::PointCloud<NormalType>());
	pcl::PointCloud <pcl::PointNormal>::Ptr model_keypoints_with_normals = pcl::PointCloud <pcl::PointNormal>::Ptr(new pcl::PointCloud <pcl::PointNormal>());
	pcl::PointCloud<DescriptorTypeSHOT>::Ptr model_descriptors_SHOT = pcl::PointCloud<DescriptorTypeSHOT>::Ptr(new pcl::PointCloud<DescriptorTypeSHOT>());
	pcl::PointCloud<RFType>::Ptr model_rf = pcl::PointCloud<RFType>::Ptr(new pcl::PointCloud<RFType>());
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

	std::string getType();
	bool loadModel();
	bool prepareModelDescriptor();
	void _3D_Matching(const PointCloudType::ConstPtr &cloud);
};


#endif