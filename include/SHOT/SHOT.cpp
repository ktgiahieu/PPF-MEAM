#include "SHOT.h"

using namespace std;
using namespace pcl;

string DescriptorSHOT::getType()
{
	return type;
}

bool DescriptorSHOT::loadModel() {

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

bool DescriptorSHOT::prepareModelDescriptor()
{
	//  Load model cloud
	if (!loadModel())
		return false;

	// ----------------------------- Set up resolution invariance -------------------------------------------------------
	float samp_rad_model = samp_rad;
	float norm_rad_model = norm_rad;
	float rf_rad_model = rf_rad_;
	float descr_rad_model = descr_rad_;
	if (use_cloud_resolution_)
	{
		float resolution = static_cast<float> (computeCloudResolution(model));
		if (resolution != 0.0f)
		{
			samp_rad_model *= resolution;
			norm_rad_model *= resolution;
			rf_rad_model *= resolution;
			descr_rad_model *= resolution;
		}

		std::cout << "Model resolution:       " << resolution << std::endl;
		std::cout << "Model sampling size:    " << samp_rad_model << std::endl;
		std::cout << "Model normal radius size:    " << norm_rad_model << std::endl;
		std::cout << "LRF support radius:     " << rf_rad_model << std::endl;
		std::cout << "SHOT descriptor radius: " << descr_rad_model << std::endl;
	}
	// --------------------------------------------- Compute Descriptors for Keypoints-------------------------------------------------------
	uniformsampling(model, samp_rad_model, model_keypoints); // Downsample Clouds to Extract keypoints
	normal(model_keypoints, 10, norm_rad_model, 'R', model_normals);//  Compute Normals

	EstimatorTypeSHOT descr_est; //  Compute Descriptor for keypoints
	descr_est.setRadiusSearch(descr_rad_model);
	descr_est.setInputCloud(model_keypoints);
	descr_est.setInputNormals(model_normals);
	descr_est.compute(*model_descriptors_SHOT);

	std::cout << "Model total points: " << model->size() << "; Selected Keypoints: " << model_keypoints->size() << std::endl;

	//  ------------------------------------------- Pre-calculate LRF ------------------------------------------------------------

	if (use_hough_) //  Using Hough3D
	{
		//  Compute (Keypoints) Reference Frames only for Hough
		pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
		rf_est.setFindHoles(true);
		rf_est.setRadiusSearch(rf_rad_model);

		rf_est.setInputCloud(model_keypoints);
		rf_est.setInputNormals(model_normals);
		rf_est.compute(*model_rf);
	}

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

	//Good keypoints
	CloudStyle goodKeypointStyle = style_violet;
	pcl::visualization::PointCloudColorHandlerCustom<PointType> model_good_keypoints_color_handler(off_model_good_kp, goodKeypointStyle.r, goodKeypointStyle.g,
		goodKeypointStyle.b);
	customViewer.viewer->addPointCloud(off_model_good_kp, model_good_keypoints_color_handler, "model_good_keypoints");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "model_good_keypoints");

	pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_good_keypoints_color_handler(scene_good_kp, goodKeypointStyle.r, goodKeypointStyle.g,
		goodKeypointStyle.b);
	customViewer.viewer->addPointCloud(scene_good_kp, scene_good_keypoints_color_handler, "scene_good_keypoints");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "scene_good_keypoints");


	return true;
}

void DescriptorSHOT::_3D_Matching(const PointCloudType::ConstPtr &cloud)
{
	mtx.lock();

	// --------------------------------------------  Preprocess--------------------------------------------------------------
	tt.tic();
	*captured_scene = *cloud;
	std::vector<float> filter_limits = { 0.0, 1.2, -0.15, 0.15, -0.1, 0.1 };
	passthrough(captured_scene, filter_limits, passthroughed_scene); // Get limited range point cloud
	sacsegmentation_extindices(passthroughed_scene, 0.008, segmented_scene); // RANSAC Segmentation and remove biggest plane (table)
	statisticalOutlinerRemoval(segmented_scene, 50, statistical_filtered_scene);// 50 k-neighbors noise removal
	*scene = *statistical_filtered_scene; // get the preprocessed scene

	// ----------------------------- Set up resolution invariance -------------------------------------------------------
	float resolution_scene = static_cast<float> (computeCloudResolution(scene));
	cout << "Resolution scene:" << resolution_scene << std::endl;
	float samp_rad_scene = samp_rad;
	float norm_rad_scene = norm_rad;
	float rf_rad_scene = rf_rad_;
	float descr_rad_scene = descr_rad_;
	float cg_size_scene = cg_size_;
	if (use_cloud_resolution_)
	{
		if (resolution_scene != 0.0f)
		{
			samp_rad_scene *= resolution_scene;
			norm_rad_scene *= resolution_scene;
			rf_rad_scene *= resolution_scene;
			descr_rad_scene *= resolution_scene;
			cg_size_scene *= resolution_scene;
		}
		std::cout << "Scene resolution:       " << resolution_scene << std::endl;
		std::cout << "Scene sampling size:    " << samp_rad_scene << std::endl;
		std::cout << "Scene normal radius size:    " << norm_rad_scene << std::endl;
		std::cout << "LRF support radius:     " << rf_rad_scene << std::endl;
		std::cout << "SHOT descriptor radius: " << descr_rad_scene << std::endl;
		std::cout << "Clustering bin size:    " << cg_size_scene << std::endl << std::endl;
	}
	// --------------------------------------------- Compute Descriptors for Keypoints-------------------------------------------------------
	uniformsampling(scene, samp_rad_scene, scene_keypoints); // Downsample Clouds to Extract keypoints
	normal(scene_keypoints, 10, norm_rad_scene, 'R', scene_normals);//  Compute Normals
	EstimatorTypeSHOT descr_est; //  Compute Descriptor for keypoints
	pcl::PointCloud<DescriptorTypeSHOT>::Ptr scene_descriptors = pcl::PointCloud<DescriptorTypeSHOT>::Ptr(new pcl::PointCloud<DescriptorTypeSHOT>());
	descr_est.setRadiusSearch(descr_rad_scene);
	descr_est.setInputCloud(scene_keypoints);
	descr_est.setInputNormals(scene_normals);
	descr_est.compute(*scene_descriptors);

	std::cout << "Scene total points: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;

	float resolution_scene_keypoints = static_cast<float> (computeCloudResolution(scene_keypoints));
	cout << "Resolution scene keypoints:" << resolution_scene_keypoints << std::endl;
	float resolution_model_keypoints = static_cast<float> (computeCloudResolution(model_keypoints));
	cout << "Resolution model keypoints:" << resolution_model_keypoints << std::endl;

	//  ------------------------------------------ Find Model-Scene Correspondences with KdTree ------------------------------------------------
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
	pcl::KdTreeFLANN<DescriptorTypeSHOT> match_search;
	match_search.setInputCloud(model_descriptors_SHOT);
	std::vector<int> model_good_keypoints_indices;
	std::vector<int> scene_good_keypoints_indices;

	//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	for (std::size_t i = 0; i < scene_descriptors->size(); ++i)
	{
		std::vector<int> neigh_indices(1);
		std::vector<float> neigh_sqr_dists(1);
		if (!std::isfinite(scene_descriptors->at(i).descriptor[0])) //skipping NaNs
			continue;
		int found_neighs = match_search.nearestKSearch(scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
		if (found_neighs == 1 && neigh_sqr_dists[0] < sqr_descr_dist)
			//  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
		{
			pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back(corr);
			model_good_keypoints_indices.push_back(corr.index_query);
			scene_good_keypoints_indices.push_back(corr.index_match);
		}
	}
	pcl::copyPointCloud(*model_keypoints, model_good_keypoints_indices, *model_good_kp);
	pcl::copyPointCloud(*scene_keypoints, scene_good_keypoints_indices, *scene_good_kp);
	std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;

	//  ------------------------------------------- Find Matching Instances ------------------------------------------------------------
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
	std::vector<pcl::Correspondences> clustered_corrs;

	if (use_hough_)
	{
		pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
		rf_est.setFindHoles(true);
		rf_est.setRadiusSearch(rf_rad_scene);

		rf_est.setInputCloud(scene_keypoints);
		rf_est.setInputNormals(scene_normals);
		rf_est.compute(*scene_rf);

		//  Clustering
		pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
		clusterer.setHoughBinSize(cg_size_scene);
		clusterer.setHoughThreshold(cg_thresh_);
		clusterer.setUseInterpolation(true);
		clusterer.setUseDistanceWeight(false);

		clusterer.setInputCloud(model_keypoints);
		clusterer.setInputRf(model_rf);
		clusterer.setSceneCloud(scene_keypoints);
		clusterer.setSceneRf(scene_rf);
		clusterer.setModelSceneCorrespondences(model_scene_corrs);
		clusterer.recognize(rototranslations, clustered_corrs);
	}
	else // Using GeometricConsistency
	{
		pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
		gc_clusterer.setGCSize(cg_size_);
		gc_clusterer.setGCThreshold(cg_thresh_);

		gc_clusterer.setInputCloud(model_keypoints);
		gc_clusterer.setSceneCloud(scene_keypoints);
		gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);

		gc_clusterer.recognize(rototranslations, clustered_corrs);
	}
	//  Output results
	std::cout << "Model instances found: " << rototranslations.size() << std::endl;
	for (std::size_t i = 0; i < rototranslations.size(); ++i)
	{
		std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
		std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;
	}

	// ------------------------------------ ICP ---------------------------------------------------------------------
	std::vector<pcl::PointCloud<PointType>::ConstPtr> instances;
	std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses
	std::vector<pcl::PointCloud<PointType>::ConstPtr> registered_instances;

	if (rototranslations.size() <= 0)
	{
		std::cout << "*** No instances found! ***" << std::endl;
	}
	else
	{
		std::cout << "Recognized Instances: " << rototranslations.size() << std::endl << std::endl;
		// Generates clouds for each instances found
		for (std::size_t i = 0; i < rototranslations.size(); ++i)
		{
			pcl::PointCloud<PointType>::Ptr rotated_model(new pcl::PointCloud<PointType>());
			pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);
			instances.push_back(rotated_model);
		}
		if (true)
		{
			std::cout << "--- ICP ---------" << std::endl;

			for (std::size_t i = 0; i < rototranslations.size(); ++i)
			{
				pcl::IterativeClosestPoint<PointType, PointType> icp;
				icp.setMaximumIterations(icp_max_iter_);
				icp.setMaxCorrespondenceDistance(icp_corr_distance_);
				icp.setInputTarget(scene);
				icp.setInputSource(instances[i]);
				pcl::PointCloud<PointType>::Ptr registered(new pcl::PointCloud<PointType>);
				icp.align(*registered);
				registered_instances.push_back(registered);
				std::cout << "Instance " << i << " ";
				if (icp.hasConverged())
				{
					std::cout << "Aligned!" << std::endl;
				}
				else
				{
					std::cout << "Not Aligned!" << std::endl;
				}
			}

			std::cout << "-----------------" << std::endl << std::endl;
		}

		// ----------------------------------------- Hypothesis Verification ---------------------------------------------------
		std::cout << "--- Hypotheses Verification ---" << std::endl;
		pcl::GlobalHypothesesVerification<PointType, PointType> GoHv;

		GoHv.setSceneCloud(scene);  // Scene Cloud
		GoHv.addModels(registered_instances, true);  //Models to verify
		GoHv.setResolution(hv_resolution_);
		GoHv.setResolutionOccupancyGrid(hv_occupancy_grid_resolution_);
		GoHv.setInlierThreshold(hv_inlier_th_);
		GoHv.setOcclusionThreshold(hv_occlusion_th_);
		GoHv.setRegularizer(hv_regularizer_);
		GoHv.setRadiusClutter(hv_rad_clutter_);
		GoHv.setClutterRegularizer(hv_clutter_reg_);
		GoHv.setDetectClutter(hv_detect_clutter_);
		GoHv.setRadiusNormals(hv_rad_normals_);

		GoHv.verify();
		GoHv.getMask(hypotheses_mask);  // i-element TRUE if hvModels[i] verifies hypotheses

		for (int i = 0; i < hypotheses_mask.size(); i++)
		{
			if (hypotheses_mask[i])
			{
				std::cout << "Instance " << i << " is GOOD! <---" << std::endl;
			}
			else
			{
				std::cout << "Instance " << i << " is bad!" << std::endl;
			}
		}
		std::cout << "-------------------------------" << std::endl;
	}

	// -------------------------------------------------------- Visualization --------------------------------------------------------
	pcl::transformPointCloud(*model_good_kp, *off_model_good_kp, Eigen::Vector3f(-0.3, 0, 0.9), Eigen::Quaternionf(1, 0, 0, 0));

	if (show_corr) {
		for (std::size_t i = 0; i < off_model_good_kp->size(); ++i)
		{
			std::stringstream ss_line;
			ss_line << "correspondence_line" << i;
			PointType& model_point = off_model_good_kp->at(i);
			PointType& scene_point = scene_good_kp->at(i);

			//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
			customViewer.viewer->addLine<PointType, PointType>(model_point, scene_point, 0, 255, 0, ss_line.str());
		}
	}

	if (!customViewer.viewer->wasStopped()) {
		customViewer.viewer->updatePointCloud(scene, "scene");
		customViewer.viewer->updatePointCloud(scene, "scene_cloud");
		customViewer.viewer->updatePointCloud(scene_keypoints, "scene_keypoints");
		customViewer.viewer->updatePointCloud(off_model_good_kp, "model_good_keypoints");
		customViewer.viewer->updatePointCloud(scene_good_kp, "scene_good_keypoints");
	}

	//Remove previous Matched model-scene 
	for (std::size_t i = 0; i < prev_instances_size; ++i)
	{
		std::stringstream ss_instance;
		ss_instance << "instance_" << i;

		if (show_instances)
			customViewer.viewer->removePointCloud(ss_instance.str(), customViewer.v2);

		ss_instance << "_registered" << std::endl;
		customViewer.viewer->removePointCloud(ss_instance.str(), customViewer.v2);
	}

	//Draw new Matched model-scene
	for (std::size_t i = 0; i < instances.size(); ++i)
	{
		std::stringstream ss_instance;
		ss_instance << "instance_" << i;

		if (show_instances) {
			CloudStyle clusterStyle = style_red;
			pcl::visualization::PointCloudColorHandlerCustom<PointType> instance_color_handler(instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
			customViewer.viewer->addPointCloud(instances[i], instance_color_handler, ss_instance.str(), customViewer.v2);
			customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());
		}
		if (rototranslations.size() > 0 && show_FP ? true : hypotheses_mask[i]) {

			CloudStyle registeredStyles = hypotheses_mask[i] ? style_green : style_cyan;
			ss_instance << "_registered" << std::endl;
			pcl::visualization::PointCloudColorHandlerCustom<PointType> registered_instance_color_handler(registered_instances[i], registeredStyles.r,
				registeredStyles.g, registeredStyles.b);
			customViewer.viewer->addPointCloud(registered_instances[i], registered_instance_color_handler, ss_instance.str(), customViewer.v2);
			customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, registeredStyles.size, ss_instance.str());
		}
	}
	prev_instances_size = instances.size();

	cout << "Thoi gian xu li: " << tt.toc() << " ms" << endl;

	// ------------------------ Clear memory --------------------------------
	passthroughed_scene->clear();
	voxelgrid_filtered_scene->clear();
	segmented_scene->clear();
	statistical_filtered_scene->clear();
	scene->clear();
	scene_normals->clear();
	scene_keypoints->clear();
	scene_descriptors->clear();
	model_good_kp->clear();
	off_model_good_kp->clear();
	scene_good_kp->clear();
	scene_rf->clear();

	mtx.unlock();
}