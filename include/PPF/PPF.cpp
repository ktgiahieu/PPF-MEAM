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
			//pcl::transformPointCloud(*cloud_xyz, *cloud_xyz, Eigen::Vector3f(0, 0, 0), Eigen::Quaternionf(0.7817, 0.1643, 0.5788, 0.1643));

			copyPointCloud(*cloud_xyz, *model_sampling);
			return true;
		}
	}
	else {
		if (file_extension == ".pcd")
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
			if (pcl::io::loadPCDFile(model_filename_, *cloud_xyz) < 0)
			{
				copyPointCloud(*cloud_xyz, *model_sampling);
				std::cout << "Error loading PCD model cloud." << std::endl;
				return false;
			}
			return true;
		}
		else if (file_extension == ".ply")
		{
			pcl::PLYReader reader;
			if (reader.read(model_filename_, *model_sampling) < 0)
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
	std::cout << "Preparing Model Descriptor Offline....." << std::endl;
	//  Load model cloud
	if (!loadModel())
		return false;

	double diameter_model = computeCloudDiameter(model_sampling);
	std::cout << "Diameter : " << diameter_model << std::endl;

	samp_rad = t_sampling * diameter_model;
	norm_rad = 2 * samp_rad;
	Lvoxel = samp_rad;
	
	//------------------------- Calculate MEAM --------------------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	copyPointCloud(*model_sampling, *cloud_xyz);

	std::vector<std::vector<float>> camera_pos(6);

	PointXYZ minPt, maxPt, avgPt;

	pcl::getMinMax3D(*cloud_xyz, minPt, maxPt);
	avgPt.x = (minPt.x + maxPt.x) / 2;
	avgPt.y = (minPt.y + maxPt.y) / 2;
	avgPt.z = (minPt.z + maxPt.z) / 2;

	float cube_length = std::max(maxPt.x - minPt.x, std::max(maxPt.y - minPt.y, maxPt.z - minPt.z));

	minPt.x = avgPt.x - cube_length;
	minPt.y = avgPt.y - cube_length;
	minPt.z = avgPt.z - cube_length;
	maxPt.x = avgPt.x + cube_length;
	maxPt.y = avgPt.y + cube_length;
	maxPt.z = avgPt.z + cube_length;

	camera_pos[0] = { avgPt.x, minPt.y, avgPt.z };
	camera_pos[1] = { maxPt.x, avgPt.y, avgPt.z };
	camera_pos[2] = { avgPt.x, maxPt.y, avgPt.z };
	camera_pos[3] = { minPt.x, avgPt.y, avgPt.z }; 
	camera_pos[4] = { avgPt.x, avgPt.y, maxPt.z };
	camera_pos[5] = { avgPt.x, avgPt.y, minPt.z };

	std::cout << "Preparing Multiview Model....." << std::endl;

	for (int i = 0; i < static_cast<int>(camera_pos.size()); ++i)
	{
		std::cout << "Preparing Viewpoint " << i << "....." << std::endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_HPR(new pcl::PointCloud<pcl::PointXYZ>());
		HPR(cloud_xyz, camera_pos[i], 3, cloud_xyz_HPR);

		*model += *cloud_xyz_HPR;
	}
	
	pcl::VoxelGrid<PointXYZ> vg;
	vg.setInputCloud(model);
	vg.setLeafSize(samp_rad, samp_rad, samp_rad);
	vg.setDownsampleAllData(false);
	vg.filter(*model_keypoints);

	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimationOMP<PointXYZ, pcl::Normal> ne;
	pcl::search::KdTree<PointXYZ>::Ptr tree(new pcl::search::KdTree<PointXYZ>);
	// Calculate all the normals of the entire surface
	ne.setInputCloud(model_keypoints);
	ne.setSearchSurface(model);
	ne.setNumberOfThreads(8);
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(norm_rad);
	ne.compute(*normals);

	pcl::concatenateFields(*model_keypoints, *normals, *model_keypoints_with_normals);


	pcl::PointCloud<pcl::PPFSignature>::Ptr descriptors_PPF = pcl::PointCloud<pcl::PPFSignature>::Ptr(new pcl::PointCloud<pcl::PPFSignature>());
	pcl::PPFEstimation<pcl::PointNormal, pcl::PointNormal, PPFSignature> ppf_estimator;
	ppf_estimator.setInputCloud(model_keypoints_with_normals);
	ppf_estimator.setInputNormals(model_keypoints_with_normals);
	ppf_estimator.compute(*descriptors_PPF);

	ppf_hashmap_search->setInputFeatureCloud(descriptors_PPF);

	//------------------------------------------------------ Visualization --------------------------------------------------------------------
	
	std::cout << "Preparing Model Visualization....." << std::endl;
	customViewer.init();
	return true;
}

void DescriptorPPF::storeLatestCloud(const PointCloudType::ConstPtr &cloud)
{
	latestCloud = cloud->makeShared();
	std::cout << "Cloud Update with Size " << latestCloud->points.size() << " ........." << std::endl;

}

void DescriptorPPF::storeLatestImage(cv::Mat& image)
{
	latestImage = image.clone();
}

void DescriptorPPF::_3D_Matching()
{
	if (latestCloud->size() == 0 || latestImage.empty())
	{
		return;
	}
	//Scene
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_scene = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr captured_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughed_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr voxelgrid_filtered_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr statistical_filtered_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_keypoints = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointNormal>::Ptr scene_keypoints_with_normals = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>());

	mtx.lock();
	pcl::copyPointCloud(*latestCloud, *captured_scene);
	pcl::copyPointCloud(*latestCloud, *colored_scene);
	cv::Mat capturedImage = latestImage.clone();
	if (use_kinect)
		cv::cvtColor(capturedImage, capturedImage, CV_RGB2BGR);
	mtx.unlock();

	std::cout << "3D Matching....." << std::endl;

	// --------------------------------------------  Preprocess--------------------------------------------------------------
	//tt.tic();
	//passthrough(captured_scene, passthrough_limits, passthroughed_scene); // Get limited range point cloud
	//cout << "passthrough in " << tt.toc() << " mseconds..." << std::endl;

	//tt.tic();
	//statisticalOutlinerRemoval(passthroughed_scene, 50, statistical_filtered_scene);// 50 k-neighbors noise removal
	//cout << "statisticalOutlinerRemoval in " << tt.toc() << " seconds..." << std::endl;

	tt.tic();
	sacsegmentation_extindices(captured_scene, 0.005, segmented_scene); // RANSAC Segmentation and remove biggest plane (table)
	cout << "sacsegmentation_extindices in " << tt.toc() << " mseconds..." << std::endl;

	tt.tic();
	*scene = *segmented_scene; // get the preprocessed scene
	cout << "copied in " << tt.toc() << " mseconds..." << std::endl;
	//pcl::io::savePLYFile( "sample.ply", *scene, true ); // Binary format

	if (scene->size() == 0)
	{
		std::cout << "No point left in scene. Skipping this frame ..." << std::endl;
		return;
	}

	tt.tic();
	// -----------------------------------------Voxel grid ------------------------------------------------------
	pcl::VoxelGrid<PointXYZ> vg_;
	vg_.setInputCloud(scene);
	vg_.setLeafSize(0.001f, 0.001f, 0.001f);
	vg_.setDownsampleAllData(true);
	vg_.filter(*scene_keypoints);
	statisticalOutlinerRemoval(scene_keypoints, 50, scene_keypoints);// 50 k-neighbors noise removal

	pcl::VoxelGrid<PointXYZ> vg;
	vg.setInputCloud(scene_keypoints);
	vg.setLeafSize(samp_rad, samp_rad, samp_rad);
	vg.setDownsampleAllData(true);
	vg.filter(*scene_keypoints);

	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimationOMP<PointXYZ, pcl::Normal> ne;
	pcl::search::KdTree<PointXYZ>::Ptr tree(new pcl::search::KdTree<PointXYZ>);
	// Calculate all the normals of the entire surface
	ne.setInputCloud(scene_keypoints);
	ne.setSearchSurface(scene);
	ne.setNumberOfThreads(8);
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(norm_rad);
	ne.compute(*normals);

	pcl::concatenateFields(*scene_keypoints, *normals, *scene_keypoints_with_normals);
	cout << "Voxel-grid filtered in " << tt.toc() << " mseconds..." << std::endl;

	tt.tic();
	// -------------------------------------------------B2B-TL MEAM ---------------------------------------------
	MyPPFRegistration<pcl::PointNormal, pcl::PointNormal> ppf_registration;
	// set parameters for the PPF registration procedure
	ppf_registration.setSceneReferencePointSamplingRate(scene_reference_point_sampling_rate);
	ppf_registration.setSceneReferredPointSamplingRate(scene_referred_point_sampling_rate);
	ppf_registration.setLvoxel(Lvoxel);
	ppf_registration.setPositionClusteringThreshold(0.03f);
	ppf_registration.setRotationClusteringThreshold(20.0f / 180.0f * float(M_PI));
	ppf_registration.setSearchMethod(ppf_hashmap_search);
	ppf_registration.setInputSource(model_keypoints_with_normals);
	ppf_registration.setInputTarget(scene_keypoints_with_normals);

	typename pcl::MyPPFRegistration<pcl::PointNormal, pcl::PointNormal>::PoseWithVotesList results;
	ppf_registration.computeFinalPoses(results);
	cout << "PPF Scene Calculation in " << tt.toc() << " mseconds..." << std::endl;









	//std::vector<PointCloud<PointXYZ>::Ptr> instances;
	//for (size_t results_i = 0; results_i < results.size(); ++results_i)
	//{
	//	pcl::PointCloud<pcl::PointXYZ>::Ptr instance(new pcl::PointCloud<pcl::PointXYZ>());
	//	pcl::transformPointCloud(*model_keypoints, *instance, results[results_i].pose);
	//	instances.push_back(instance);
	//}
	
	tt.tic();
	//----------------------------------------------------------- ICP ----------------------------------------------------
	std::vector<PointCloud<PointXYZ>::ConstPtr> instances;
	for (size_t results_i = 0; results_i < results.size(); ++results_i)
	{
		// Generates clouds for each instances found
		pcl::PointCloud<pcl::PointXYZ>::Ptr instance(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::transformPointCloud(*model, *instance, results[results_i].pose);

		std::vector<float> camera_pos = { 0, 0 ,0 };
		HPR(instance, camera_pos, 3, instance);

		pcl::VoxelGrid<PointXYZ> vg_icp;
		vg_icp.setInputCloud(instance);
		vg_icp.setLeafSize(samp_rad, samp_rad, samp_rad);
		vg_icp.setDownsampleAllData(true);
		vg_icp.filter(*instance);

		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setMaximumIterations(icp_max_iter_);
		icp.setMaxCorrespondenceDistance(icp_corr_distance_);
		icp.setInputTarget(scene_keypoints);
		icp.setInputSource(instance);
		pcl::PointCloud<pcl::PointXYZ>::Ptr registered_instance(new pcl::PointCloud<pcl::PointXYZ>);
		icp.align(*registered_instance);

		std::cout << "Instance " << results_i << " ";
		if (icp.hasConverged())
		{
			std::cout << "Aligned!" << std::endl;
		}
		else
		{
			std::cout << "Not Aligned!" << std::endl;
		}

		Eigen::Matrix4f transformation = icp.getFinalTransformation();
		transformation = transformation * results[results_i].pose.matrix();

		//pcl::transformPointCloud(*model_keypoints, *instance, transformation);
		//instances.push_back(instance);
		instances.push_back(registered_instance);
	}
	cout << "ICP in " << tt.toc() << " mseconds..." << std::endl;
	cout << "instances size: " << instances.size()	 << std::endl;



	std::vector<bool> hypotheses_mask = { true, true, true, true, true, true, true };
	//tt.tic();
	//// ----------------------------------------- Hypothesis Verification ---------------------------------------------------
	//std::cout << "--- Hypotheses Verification ---" << std::endl;
	//std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses
	//pcl::GlobalHypothesesVerification<PointXYZ, PointXYZ> GoHv;

	//GoHv.setSceneCloud(scene);
	//GoHv.addModels(instances, true);  //Models to verify
	//GoHv.setResolution(hv_resolution_);
	//GoHv.setResolutionOccupancyGrid(hv_occupancy_grid_resolution_);
	//GoHv.setInlierThreshold(hv_inlier_th_);
	//GoHv.setOcclusionThreshold(hv_occlusion_th_);
	//GoHv.setRegularizer(hv_regularizer_);
	//GoHv.setRadiusClutter(hv_rad_clutter_);
	//GoHv.setClutterRegularizer(hv_clutter_reg_);
	//GoHv.setDetectClutter(hv_detect_clutter_);
	//GoHv.setRadiusNormals(hv_rad_normals_);

	//GoHv.verify();
	//GoHv.getMask(hypotheses_mask);  // i-element TRUE if hvModels[i] verifies hypotheses

	//for (int i = 0; i < hypotheses_mask.size(); i++)
	//{
	//	if (hypotheses_mask[i] && static_cast<float>(results[i].votes) > 0.7f * static_cast<float>(results[0].votes))
	//	{
	//		std::cout << "Instance " << i << " is GOOD! <---" << std::endl;
	//		std::cout << "Instance " << i << " score: " << results[i].votes << std::endl;
	//		std::cout << "Max score: " << results[0].votes << std::endl;
	//	}
	//	else
	//		hypotheses_mask[i] = false;
	//}


	//std::cout << "-------------------------------" << std::endl;
	//cout << "HV in " << tt.toc() << " mseconds..." << std::endl;

	tt.tic();
	// -------------------------------------------------------- Visualization --------------------------------------------------------
	if (!customViewer.viewer->wasStopped()) {
		customViewer.viewer->removeAllShapes();
		customViewer.viewer->removeAllPointClouds();



		//customViewer.viewer->addPointCloud<pcl::PointXYZ>(scene, "segmented_cloud");

		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(colored_scene);
		customViewer.viewer->addPointCloud<pcl::PointXYZRGB>(colored_scene, rgb, "colored_scene");
		customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "colored_scene");
		

		//customViewer.viewer->addPointCloud<pcl::PointXYZ>(scene_keypoints, "scene_keypoints");
		//customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_keypoints");
		//customViewer.viewer->addPointCloudNormals<PointXYZTangent, PointXYZTangent>(scene_keypoints_with_normals, scene_keypoints_with_normals, 1, 0.005, "scene_keypoints_with_normals");
		std::cout << "2" << std::endl;

		//Draw new Matched model-scene
		for (std::size_t i = 0; i < instances.size(); ++i)
		{
			std::stringstream ss_instance;
			ss_instance << "instance_" << i;
			if (show_FP ? true : hypotheses_mask[i])
			{
				CloudStyle clusterStyle = hypotheses_mask[i] ? style_green : style_red;
				pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> instance_color_handler(instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
				customViewer.viewer->addPointCloud(instances[i], instance_color_handler, ss_instance.str());
				customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());
			}


			//CloudStyle clusterStyle = style_red;
			//pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> instance_color_handler(instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
			//customViewer.viewer->addPointCloud(instances[i], instance_color_handler, ss_instance.str());
			//customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());

		}
		
	}
	cout << "Visualize in " << tt.toc() << " mseconds..." << std::endl;
}