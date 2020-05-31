#ifndef PPF
#define PPF

#include "pclFunction.h"
#include "meshSampling.h"

class DescriptorPPF
{
private:
	//IO
	std::string model_filename_ = "../../../data/nap.STL";
	bool is_mesh = true;

	//Algorithm params
	bool show_keypoints_ = true;
	bool show_corr = false;
	bool show_instances = false;
	bool use_cloud_resolution_ = false;
	bool use_hough_ = true;
	float samp_rad = 0.01f;
	float norm_rad = 2.5 * samp_rad;
	float descr_rad_ = 1.0f * norm_rad;

	std::string type = "PPF";

	// Model 
	PointCloudType::Ptr model = PointCloudType::Ptr(new PointCloudType());
	PointCloudType::Ptr model_keypoints = PointCloudType::Ptr(new PointCloudType());

	pcl::PointCloud<NormalType>::Ptr model_normals = pcl::PointCloud<NormalType>::Ptr(new pcl::PointCloud<NormalType>());
	pcl::PointCloud <pcl::PointNormal>::Ptr model_keypoints_with_normals = pcl::PointCloud <pcl::PointNormal>::Ptr(new pcl::PointCloud <pcl::PointNormal>());
	pcl::PointCloud<DescriptorTypePPF>::Ptr model_descriptors_PPF = pcl::PointCloud<DescriptorTypePPF>::Ptr(new pcl::PointCloud<DescriptorTypePPF>());
	pcl::PPFHashMapSearch::Ptr ppf_hashmap_search = pcl::PPFHashMapSearch::Ptr(new pcl::PPFHashMapSearch(12.0f / 180.0f * float(M_PI), 0.05f));
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