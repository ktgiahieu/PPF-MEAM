#include "PPF.h"

using namespace std;
using namespace pcl;

string DescriptorPPF::getType()
{
	return type;
}

bool DescriptorPPF::loadModel() {

	string file_extension = model_filename_.substr(model_filename_.find_last_of('.'));
	if (is_mesh) {
		if (file_extension == ".ply" || file_extension == ".obj" || file_extension == ".stl" || file_extension == ".STL")
		{
			std::cout << "Loading mesh..." << std::endl;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
			meshSampling(model_filename_, 1000000, 0.0005f, false, cloud_xyz);
			copyPointCloud(*cloud_xyz, *model);
			return true;
		}
	}
	else {
		if (file_extension == ".pcd")
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
			if (pcl::io::loadPCDFile(model_filename_, *cloud_xyz) < 0)
			{
				std::cout << "Error loading PCD model cloud." << std::endl;
				return false;
			}
			return true;
		}
		else if (file_extension == ".ply")
		{
			pcl::PLYReader reader;
			if (reader.read(model_filename_, *model) < 0)
			{
				std::cout << "Error loading PLY model cloud." << std::endl;
				return false;
			}
			return true;
		}
	}
	std::cout << "No file name found." << std::endl;
	return false;
}

bool DescriptorPPF::prepareModelDescriptor()
{
	//  Load model cloud
	if (!loadModel())
		return false;

	// ----------------------------- Set up resolution invariance -------------------------------------------------------
	float samp_rad_model = samp_rad;
	float norm_rad_model = norm_rad;
	float descr_rad_model = descr_rad_;
	if (use_cloud_resolution_)
	{
		float resolution = static_cast<float> (computeCloudResolution(model));
		if (resolution != 0.0f)
		{
			samp_rad_model *= resolution;
			norm_rad_model *= resolution;
			descr_rad_model *= resolution;
		}

		std::cout << "Model resolution:       " << resolution << std::endl;
		std::cout << "Model sampling size:    " << samp_rad_model << std::endl;
		std::cout << "Model normal radius size:    " << norm_rad_model << std::endl;
		std::cout << "SHOT descriptor radius: " << descr_rad_model << std::endl;
	}
	// --------------------------------------------- Compute Descriptors for Keypoints-------------------------------------------------------

	voxelgrid(model, samp_rad_model, model_keypoints); // Downsample Clouds to Extract keypoints
	normal(model_keypoints, 10, norm_rad_model, 'R', model_normals);//  Compute Normals
	pcl::PointCloud <pcl::PointXYZ>::Ptr model_keypoints_XYZ = pcl::PointCloud <pcl::PointXYZ>::Ptr(new pcl::PointCloud <pcl::PointXYZ>());
	copyPointCloud(*model_keypoints, *model_keypoints_XYZ);
	pcl::concatenateFields(*model_keypoints_XYZ, *model_normals, *model_keypoints_with_normals);

	PPFEstimation<PointNormal, PointNormal, PPFSignature> ppf_estimator;
	ppf_estimator.setInputCloud(model_keypoints_with_normals);
	ppf_estimator.setInputNormals(model_keypoints_with_normals);
	ppf_estimator.compute(*model_descriptors_PPF);

	ppf_hashmap_search->setInputFeatureCloud(model_descriptors_PPF);

	std::cout << "Model total points: " << model->size() << "; Selected Keypoints: " << model_keypoints->size() << std::endl;

	//------------------------------------------------------ Visualization --------------------------------------------------------------------
	customViewer.init();
	pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(scene);
	customViewer.viewer->addPointCloud<PointType>(scene, rgb, "scene", customViewer.v1);
	customViewer.viewer->addPointCloud<PointType>(scene, rgb, "scene_cloud", customViewer.v2);
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene_cloud");

	//Uniform keypoints
	CloudStyle uniformKeyPointStyle = style_white;

	pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler(scene_keypoints, uniformKeyPointStyle.r, uniformKeyPointStyle.g, uniformKeyPointStyle.b);
	customViewer.viewer->addPointCloud(scene_keypoints, scene_keypoints_color_handler, "scene_keypoints", customViewer.v2);
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

	pcl::transformPointCloud(*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-0.3, 0, 0.9), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler(off_scene_model_keypoints, uniformKeyPointStyle.r, uniformKeyPointStyle.g, uniformKeyPointStyle.b);
	customViewer.viewer->addPointCloud(off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints", customViewer.v2);
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");

	return true;
}

void DescriptorPPF::_3D_Matching(const PointCloudType::ConstPtr &cloud)
{
	mtx.lock();

	// --------------------------------------------  Preprocess--------------------------------------------------------------
	tt.tic();
	*captured_scene = *cloud;
	std::vector<float> filter_limits = { 0.0, 1.2, -0.15, 0.15, -0.12, 0.12 };
	passthrough(captured_scene, filter_limits, passthroughed_scene); // Get limited range point cloud
	sacsegmentation_extindices(passthroughed_scene, 0.008, segmented_scene); // RANSAC Segmentation and remove biggest plane (table)
	statisticalOutlinerRemoval(segmented_scene, 50, statistical_filtered_scene);// 50 k-neighbors noise removal
	*scene = *statistical_filtered_scene; // get the preprocessed scene
	// ----------------------------- Set up resolution invariance -------------------------------------------------------
	float resolution_scene = static_cast<float> (computeCloudResolution(scene));
	cout << "Resolution scene:" << resolution_scene << std::endl;
	float samp_rad_scene = samp_rad;
	float norm_rad_scene = norm_rad;
	float descr_rad_scene = descr_rad_;
	if (use_cloud_resolution_)
	{
		if (resolution_scene != 0.0f)
		{
			samp_rad_scene *= resolution_scene;
			norm_rad_scene *= resolution_scene;
			descr_rad_scene *= resolution_scene;
		}
		std::cout << "Scene resolution:       " << resolution_scene << std::endl;
		std::cout << "Scene sampling size:    " << samp_rad_scene << std::endl;
		std::cout << "Scene normal radius size:    " << norm_rad_scene << std::endl;
		std::cout << "SHOT descriptor radius: " << descr_rad_scene << std::endl;
	}
	// --------------------------------------------- Downsample and Calculate Normals -------------------------------------------------------

	voxelgrid(scene, samp_rad_scene, scene_keypoints); // Downsample Clouds to Extract keypoints
	normal(scene_keypoints, 10, norm_rad_scene, 'R', scene_normals);//  Compute Normals
	pcl::PointCloud <pcl::PointXYZ>::Ptr scene_keypoints_XYZ = pcl::PointCloud <pcl::PointXYZ>::Ptr(new pcl::PointCloud <pcl::PointXYZ>());
	copyPointCloud(*scene_keypoints, *scene_keypoints_XYZ);
	pcl::PointCloud <pcl::PointNormal>::Ptr scene_keypoints_with_normals = pcl::PointCloud <pcl::PointNormal>::Ptr(new pcl::PointCloud <pcl::PointNormal>());

	pcl::concatenateFields(*scene_keypoints_XYZ, *scene_normals, *scene_keypoints_with_normals);
	std::cout << "Scene total points: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;

	float resolution_scene_keypoints = static_cast<float> (computeCloudResolution(scene_keypoints));
	cout << "Resolution scene keypoints:" << resolution_scene_keypoints << std::endl;
	float resolution_model_keypoints = static_cast<float> (computeCloudResolution(model_keypoints));
	cout << "Resolution model keypoints:" << resolution_model_keypoints << std::endl;

	PPFRegistration<PointNormal, PointNormal> ppf_registration;
	// set parameters for the PPF registration procedure
	ppf_registration.setSceneReferencePointSamplingRate(10);
	ppf_registration.setPositionClusteringThreshold(0.2f);
	ppf_registration.setRotationClusteringThreshold(30.0f / 180.0f * float(M_PI));
	ppf_registration.setSearchMethod(ppf_hashmap_search);
	ppf_registration.setInputSource(model_keypoints_with_normals);
	ppf_registration.setInputTarget(scene_keypoints_with_normals);

	PointCloud<PointNormal> cloud_output_subsampled;
	ppf_registration.align(cloud_output_subsampled);

	Eigen::Matrix4f mat = ppf_registration.getFinalTransformation();
	Eigen::Affine3f final_transformation(mat);

	std::vector<PointCloudType::Ptr> instances;
	PointCloudType::Ptr instance(new PointCloudType());
	pcl::transformPointCloud(*model_keypoints, *instance, final_transformation);
	instances.push_back(instance);

	// -------------------------------------------------------- Visualization --------------------------------------------------------

	if (!customViewer.viewer->wasStopped()) {
		customViewer.viewer->updatePointCloud(scene, "scene");
		customViewer.viewer->updatePointCloud(scene, "scene_cloud");
		customViewer.viewer->updatePointCloud(scene_keypoints, "scene_keypoints");
	}

	//Remove previous Matched model-scene 
	for (std::size_t i = 0; i < prev_instances_size; ++i)
	{
		std::stringstream ss_instance;
		ss_instance << "instance_" << i;

		customViewer.viewer->removePointCloud(ss_instance.str(), customViewer.v2);
	}

	//Draw new Matched model-scene
	for (std::size_t i = 0; i < instances.size(); ++i)
	{
		std::stringstream ss_instance;
		ss_instance << "instance_" << i;

		CloudStyle clusterStyle = style_red;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> instance_color_handler(instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
		customViewer.viewer->addPointCloud(instances[i], instance_color_handler, ss_instance.str(), customViewer.v2);
		customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());
	}
	prev_instances_size = instances.size();

	cout << "Thoi gian xu li: " << tt.toc() << " ms" << endl;

	// ------------------------ Clear memory --------------------------------
	captured_scene->clear();
	passthroughed_scene->clear();
	voxelgrid_filtered_scene->clear();
	segmented_scene->clear();
	statistical_filtered_scene->clear();
	scene->clear();
	scene_normals->clear();
	scene_keypoints_with_normals->clear();
	scene_keypoints->clear();

	mtx.unlock();
}