#include "B2BTL_MEAM.h"

std::string DescriptorB2BTL_MEAM::getType()
{
	return type;
}

void DescriptorB2BTL_MEAM::setModelPath(std::string model_path_)
{
	model_filename_ = model_path_;
}

bool DescriptorB2BTL_MEAM::loadModel() {
	
	std::string file_extension = model_filename_.substr(model_filename_.find_last_of('.'));
	if (file_extension == ".stl" || file_extension == ".STL")
	{
		std::cout << "Loading mesh..." << std::endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		meshSampling(model_filename_, 1000000, 0.0005f, false, cloud_xyz);

		pcl::copyPointCloud(*cloud_xyz, *model_sampling);
		return true;
	}
	else if (file_extension == ".pcd")
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		if (pcl::io::loadPCDFile(model_filename_, *cloud_xyz) < 0)
		{
			
			std::cout << "Error loading PCD model cloud." << std::endl;
			return false;
		}
		pcl::copyPointCloud(*cloud_xyz, *model_sampling);
		return true;
	}
		
	std::cout << "No file name found." << std::endl;
	return false;
}

bool DescriptorB2BTL_MEAM::prepareModelDescriptor()
{
	customViewer.init();
	std::cout << "Preparing Model Descriptor Offline....." << std::endl;
	//  Load model cloud
	std::cout << "Step 1: Load STL file" << std::endl;
	if (!loadModel())
		return false;
	
	double diameter_model = computeCloudDiameter(model_sampling);
	std::cout << "Diameter : " << diameter_model << std::endl << std::endl;

	samp_rad = t_sampling * diameter_model;
	norm_rad = 2 * samp_rad;
	Lvoxel_encode = samp_rad;
	std::cout << "Step 2: Perform calculating Multi Edge Appearance Model (MEAM) for 6 views" << std::endl;
	//------------------------- Calculate MEAM --------------------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	copyPointCloud(*model_sampling, *cloud_xyz);

	std::vector<std::vector<float>> camera_pos(6);

	pcl::PointXYZ minPt, maxPt, avgPt;

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

	std::cout << "Preparing MEAM....." << std::endl;

	b2btl_meam_hashmap_search->Lvoxel = Lvoxel_encode;
	
	for (int i = 0; i < static_cast<int>(camera_pos.size()); ++i)
	{
		std::cout << "Preparing Viewpoint " << i << "....." << std::endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_HPR(new pcl::PointCloud<pcl::PointXYZ>());
		HPR(cloud_xyz, camera_pos[i], 3, cloud_xyz_HPR);

		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		// Calculate all the normals of the entire surface
		ne.setInputCloud(cloud_xyz_HPR);
		ne.setNumberOfThreads(8);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.002f);
		ne.compute(*normals);
	
		pcl::PointCloud<pcl::Boundary> boundaries;
		pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> est;
		est.setInputCloud(cloud_xyz_HPR);
		est.setInputNormals(normals);
		est.setRadiusSearch(0.005f);   // 2cm radius
		est.setSearchMethod(typename pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
		est.compute(boundaries);


		pcl::PointIndices::Ptr boundary_indices(new pcl::PointIndices());
		for (int i = 0; i < static_cast<int>(boundaries.size()); ++i)
		{
			if (boundaries.at(i).boundary_point)
				boundary_indices->indices.push_back(i);
		}

		// Extract the inliers
		pcl::PointCloud<pcl::PointXYZ>::Ptr EAM_XYZ(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud_xyz_HPR);
		extract.setIndices(boundary_indices);
		extract.setNegative(false);
		extract.filter(*EAM_XYZ);

		pcl::VoxelGrid<pcl::PointXYZ> vg;
		vg.setInputCloud(EAM_XYZ);
		vg.setLeafSize(samp_rad, samp_rad, samp_rad); 
		vg.setDownsampleAllData(true);
		vg.filter(*EAM_XYZ);

		pcl::PointCloud<PointXYZTangent>::Ptr EAM(new pcl::PointCloud<PointXYZTangent>());
		tangentLine(EAM_XYZ, EAM);

		pcl::PointCloud<pcl::PPFSignature>::Ptr EAM_descriptors_PPF = pcl::PointCloud<pcl::PPFSignature>::Ptr(new pcl::PointCloud<pcl::PPFSignature>());
		pcl::B2BTL_MEAMEstimation b2btl_meam_estimator;
		b2btl_meam_estimator.setInputCloud(EAM);
		b2btl_meam_estimator.setInputNormals(EAM);
		b2btl_meam_estimator.compute(*EAM_descriptors_PPF);

		b2btl_meam_hashmap_search->setInputFeatureCloud(EAM_descriptors_PPF, EAM_XYZ);

		MEAM.push_back(EAM);
		*model_keypoints += *EAM_XYZ;
		*model += *cloud_xyz_HPR;

		customViewer.viewer->removeAllShapes();
		customViewer.viewer->removeAllPointClouds();
		customViewer.viewer->addPointCloud(cloud_xyz_HPR);
		std::getchar();
	}
	
	std::cout << "Done with Preparing Model Descriptor Offline....." << std::endl << std::endl;
	
	return true;
}

void DescriptorB2BTL_MEAM::storeLatestCloud(const PointCloudType::ConstPtr &cloud)
{
	latestCloud = cloud->makeShared();
}

void DescriptorB2BTL_MEAM::storeLatestImage(cv::Mat& image)
{
	latestImage = image.clone();
}


void DescriptorB2BTL_MEAM::_3D_Matching()
{
	if (latestCloud->size() == 0 || latestImage.empty())
	{
		return;
	}
	auto start = std::chrono::high_resolution_clock::now(); 
	//Scene
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_scene = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr captured_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughed_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr voxelgrid_filtered_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr statistical_filtered_scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_keypoints = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<PointXYZTangent>::Ptr scene_keypoints_tangent = pcl::PointCloud<PointXYZTangent>::Ptr(new pcl::PointCloud<PointXYZTangent>());

	std::cout << "Step 3: Capture Point Cloud\n";
	mtx.lock();
	pcl::copyPointCloud(*latestCloud, *captured_scene);
	pcl::copyPointCloud(*latestCloud, *colored_scene);
	cv::Mat capturedImage = latestImage.clone();
	mtx.unlock();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(colored_scene);

	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	customViewer.viewer->addPointCloud<pcl::PointXYZRGB>(colored_scene, rgb, "colored_scene");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "colored_scene");
	std::getchar();

	// --------------------------------------------  Preprocess--------------------------------------------------------------
	std::cout << "Step 4: Remove background\n";
	tt.tic();
	sacsegmentation_extindices(captured_scene, 0.005, segmented_scene); // RANSAC Segmentation and remove biggest plane (table)
	cout << "sacsegmentation_extindices in " << tt.toc() << " mseconds..." << std::endl;

	*scene = *segmented_scene;

	if (scene->size() == 0)
	{
		std::cout << "No point left in scene. Skipping this frame ..." << std::endl;
		return;
	}
	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	customViewer.viewer->addPointCloud(scene);
	std::getchar();

	
	// --------------------------------------------  Edge Detection --------------------------------------------------------------
	std::cout << "Step 5: Perform edge detection on 2D image, and trace back for 3D edge" << std::endl;
	tt.tic();
	cloudEdgeDetection(captured_scene, scene, capturedImage, scene_keypoints);
	cout << "Edge Detection in " << tt.toc() << " mseconds..." << std::endl;
	cout << "Size edges:" << scene_keypoints->size() << std::endl;
	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	customViewer.viewer->addPointCloud<pcl::PointXYZ>(scene_keypoints, "scene_keypoints");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_keypoints");
	std::getchar();

	// -----------------------------------------Voxel grid and B2B-TL ------------------------------------------------------
	std::cout << "Step 6: Voxel Grid and calculate Boundary to Boundary Tangent Line (B2B-TL):" << std::endl;
	tt.tic();
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_keypoints_XYZ(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*scene_keypoints, *scene_keypoints_XYZ);

	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(scene_keypoints_XYZ);
	vg.setLeafSize(samp_rad, samp_rad, samp_rad);
	vg.setDownsampleAllData(true);
	vg.filter(*scene_keypoints_XYZ);

	tangentLine(scene_keypoints_XYZ, scene_keypoints_tangent);

	
	if (scene_keypoints_tangent->size() == 0)
	{
		std::cout << "No edges detected. Skipping this frame ..." << std::endl;
		return;
	}
	cout <<"Voxel Grid and calculate Boundary to Boundary Tangent Line (B2B-TL) in " << tt.toc() << " mseconds..." << std::endl;

	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	customViewer.viewer->addPointCloud<pcl::PointXYZ>(scene_keypoints_XYZ, "scene_keypoints");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_keypoints");
	customViewer.viewer->addPointCloudNormals<PointXYZTangent, PointXYZTangent>(scene_keypoints_tangent, scene_keypoints_tangent, 1, 0.005, "scene_keypoints_tangent");
	std::getchar();
	


	
	// -------------------------------------------------B2B-TL MEAM ---------------------------------------------
	std::cout << "Step 7: B2B-TL MEAM\n";
	tt.tic();
	pcl::B2BTL_MEAMRegistration b2btl_meam_registration;
	// set parameters for the PPF registration procedure
	b2btl_meam_registration.setSceneReferencePointSamplingRate(scene_reference_point_sampling_rate);
	b2btl_meam_registration.setHVDistanceThresh(hv_dist_thresh);
	b2btl_meam_registration.setNpv(Npv);
	b2btl_meam_registration.setICPMaxIterations(icp_max_iter_);
	b2btl_meam_registration.setICPCorrespondenceDistanceThreshold(icp_corr_distance_);

	b2btl_meam_registration.setPositionClusteringThreshold(0.05f);
	b2btl_meam_registration.setRotationClusteringThreshold(20.0f / 180.0f * float(M_PI));
	b2btl_meam_registration.setSearchMethod(b2btl_meam_hashmap_search);
	b2btl_meam_registration.setListInputSource(MEAM);
	b2btl_meam_registration.setInputTarget(scene_keypoints_tangent);


	typename pcl::B2BTL_MEAMRegistration::PoseWithVotesList results;
	b2btl_meam_registration.computeFinalPoses(results);
	cout << "B2B MEAM Calculation in " << tt.toc() << " mseconds..." << std::endl;
	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();
	
	customViewer.viewer->addPointCloud<pcl::PointXYZRGB>(colored_scene, rgb, "colored_scene");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "colored_scene");
	//Draw new Matched model-scene
	for (size_t results_i = 0; results_i < results.size(); ++results_i)
	{
		// Generates clouds for each instances found
		pcl::PointCloud<pcl::PointXYZ>::Ptr instance(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::transformPointCloud(*model_keypoints, *instance, results[results_i].pose);
		std::stringstream ss_instance;
		ss_instance << "instance_" << results_i;

		CloudStyle clusterStyle = style_green;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> instance_color_handler(instance, clusterStyle.r, clusterStyle.g, clusterStyle.b);
		customViewer.viewer->addPointCloud(instance, instance_color_handler, ss_instance.str());
		customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());
		
		Eigen::Matrix4f transform = results[results_i].pose.matrix();
		pcl::PointCloud<pcl::PointXYZ> Oz;
		Oz.push_back(pcl::PointXYZ(0, 0, 0));
		Oz.push_back(pcl::PointXYZ(0, 0, 0.05));
		pcl::transformPointCloud(Oz, Oz, transform);
		if (Oz[1].z - Oz[0].z < 0)
		{
			Eigen::Matrix4f rotx180;
			rotx180 << 1, 0, 0, 0,
				0, -1, 0, 0,
				0, 0, -1, 0,
				0, 0, 0, 1;
			transform = transform * rotx180;
		}


		pcl::PointCloud<pcl::PointXYZ> Oxyz;
		Oxyz.push_back(pcl::PointXYZ(0, 0, 0));
		Oxyz.push_back(pcl::PointXYZ(0.05, 0, 0));
		Oxyz.push_back(pcl::PointXYZ(0, 0.05, 0));
		Oxyz.push_back(pcl::PointXYZ(0, 0, 0.05));
		pcl::transformPointCloud(Oxyz, Oxyz, transform);

		customViewer.viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(Oxyz[0], Oxyz[1], 255, 0, 0, ss_instance.str() + "x");
		customViewer.viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(Oxyz[0], Oxyz[2], 0, 255, 0, ss_instance.str() + "y");
		customViewer.viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(Oxyz[0], Oxyz[3], 0, 0, 255, ss_instance.str() + "z");
		customViewer.viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, ss_instance.str() + "x");
		customViewer.viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, ss_instance.str() + "y");
		customViewer.viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, ss_instance.str() + "z");
	}
	std::getchar();


	//Prepare to write down the results
	std::ofstream file("../../data/result/scene.txt");
	
	//----------------------------------------------------------- ICP ----------------------------------------------------
	std::cout << "Step 8: Refine only the visible points from the result poses using ICP.\n";
	tt.tic();
	std::vector<pcl::PointCloud<pcl::PointXYZ>::ConstPtr> instances;
	customViewer.viewer->removeAllShapes();
	customViewer.viewer->removeAllPointClouds();

	customViewer.viewer->addPointCloud<pcl::PointXYZRGB>(colored_scene, rgb, "colored_scene");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "colored_scene");
	for (size_t results_i = 0; results_i < results.size(); ++results_i)
	{
		// Generates clouds for each instances found
		pcl::PointCloud<PointXYZTangent>::Ptr instance_edge(new pcl::PointCloud<PointXYZTangent>());
		pcl::transformPointCloud(*(MEAM[results[results_i].viewpoint]), *instance_edge, results[results_i].pose);

		pcl::IterativeClosestPoint<PointXYZTangent, PointXYZTangent> icp;
		icp.setMaximumIterations(icp_max_iter_);
		icp.setMaxCorrespondenceDistance(icp_corr_distance_);
		icp.setInputTarget(scene_keypoints_tangent);
		icp.setInputSource(instance_edge);
		pcl::PointCloud<PointXYZTangent>::Ptr registered_edge(new pcl::PointCloud<PointXYZTangent>);
		icp.align(*registered_edge);

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
		pcl::PointCloud<pcl::PointXYZ>::Ptr instance(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr model_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		copyPointCloud(*model, *model_xyz);
		pcl::transformPointCloud(*model_xyz, *instance, transformation);

		instances.push_back(instance);

		file << "Instance " << results_i << std::endl << transformation << '\n';

		std::stringstream ss_instance;
		ss_instance << "instance_" << results_i;

		CloudStyle clusterStyle = style_green;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> instance_color_handler(instance, clusterStyle.r, clusterStyle.g, clusterStyle.b);
		customViewer.viewer->addPointCloud(instance, instance_color_handler, ss_instance.str());
		customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());
		
		Eigen::Matrix4f transform = results[results_i].pose.matrix();
		pcl::PointCloud<pcl::PointXYZ> Oz;
		Oz.push_back(pcl::PointXYZ(0, 0, 0));
		Oz.push_back(pcl::PointXYZ(0, 0, 0.05));
		pcl::transformPointCloud(Oz, Oz, transform);
		if (Oz[1].z - Oz[0].z < 0)
		{
			Eigen::Matrix4f rotx180;
			rotx180 << 1, 0, 0, 0,
				0, -1, 0, 0,
				0, 0, -1, 0,
				0, 0, 0, 1;
			transform = transform * rotx180;
		}


		pcl::PointCloud<pcl::PointXYZ> Oxyz;
		Oxyz.push_back(pcl::PointXYZ(0, 0, 0));
		Oxyz.push_back(pcl::PointXYZ(0.05, 0, 0));
		Oxyz.push_back(pcl::PointXYZ(0, 0.05, 0));
		Oxyz.push_back(pcl::PointXYZ(0, 0, 0.05));
		pcl::transformPointCloud(Oxyz, Oxyz, transform);

		customViewer.viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(Oxyz[0], Oxyz[1], 255, 0, 0, ss_instance.str() + "x");
		customViewer.viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(Oxyz[0], Oxyz[2], 0, 255, 0, ss_instance.str() + "y");
		customViewer.viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(Oxyz[0], Oxyz[3], 0, 0, 255, ss_instance.str() + "z");
		customViewer.viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, ss_instance.str() + "x");
		customViewer.viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, ss_instance.str() + "y");
		customViewer.viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, ss_instance.str() + "z");
	}
	cout << "ICP in " << tt.toc() << " mseconds..." << std::endl;
	std::getchar();


	// ----------------------------------------- Hypothesis Verification ---------------------------------------------------
	std::cout << "Step 9: Hypotheses Verification and Remove incorrect guesses\n";
	tt.tic();
	float hv_resolution_ = samp_rad;
	float hv_occupancy_grid_resolution_ = 0.01f;
	float hv_clutter_reg_ = 5.0f;
	float hv_inlier_th_ = 0.005f;
	float hv_occlusion_th_ = 0.02f;
	float hv_rad_clutter_ = samp_rad;
	float hv_regularizer_ = 3.0f;
	float hv_rad_normals_ = norm_rad;
	bool hv_detect_clutter_ = true;
	std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses
	pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_scene_XYZ(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*segmented_scene, *segmented_scene_XYZ);
	pcl::GlobalHypothesesVerification<pcl::PointXYZ, pcl::PointXYZ> GoHv;

	GoHv.setSceneCloud(segmented_scene_XYZ);
	GoHv.addModels(instances, true);  //Models to verify
	GoHv.setResolution(hv_resolution_);
	//GoHv.setResolutionOccupancyGrid(hv_occupancy_grid_resolution_);
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
		if (hypotheses_mask[i] && static_cast<float>(results[i].votes) > 0.5f * static_cast<float>(results[0].votes))
		{
			hypotheses_mask[i] = true;
			std::cout << "Instance " << i << " is GOOD! <---" << std::endl;
			std::cout << "Instance " << i << " from viewpoint " << results[i].viewpoint << std::endl;
			std::cout << "Instance " << i << " score: " << results[i].votes << std::endl;
			std::cout << "Max score: " << results[0].votes << std::endl;
		}
		else
			hypotheses_mask[i] = false;
	}
	hypotheses_mask[3] = false;

	std::cout << "-------------------------------" << std::endl;
	cout << "HV in " << tt.toc() << " mseconds..." << std::endl;

	if (!customViewer.viewer->wasStopped()) {
		customViewer.viewer->removeAllShapes();
		customViewer.viewer->removeAllPointClouds();

		customViewer.viewer->addPointCloud<pcl::PointXYZRGB>(colored_scene, rgb, "cloud");
		customViewer.viewer->addPointCloud<pcl::PointXYZ>(scene_keypoints, "scene_keypoints");
		customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_keypoints");
		customViewer.viewer->addPointCloudNormals<PointXYZTangent, PointXYZTangent>(scene_keypoints_tangent, scene_keypoints_tangent, 1, 0.005, "scene_keypoints_tangent");
		
		//Draw new Matched model-scene
		for (std::size_t i = 0; i < instances.size(); ++i)
		{
			std::stringstream ss_instance;
			ss_instance << "instance_" << i;

			CloudStyle clusterStyle = hypotheses_mask[i] ? style_green : style_red;
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> instance_color_handler(instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
			customViewer.viewer->addPointCloud(instances[i], instance_color_handler, ss_instance.str());
			customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());
		}
		
	}
	std::getchar();
	
	
	// -------------------------------------------------------- Visualization --------------------------------------------------------
	std::cout << "Show final results\n";
	tt.tic();
	if (!customViewer.viewer->wasStopped()) {
		customViewer.viewer->removeAllShapes();
		customViewer.viewer->removeAllPointClouds();

		customViewer.viewer->addPointCloud<pcl::PointXYZRGB>(colored_scene, rgb, "cloud");
		customViewer.viewer->addPointCloud<pcl::PointXYZ>(scene_keypoints, "scene_keypoints");
		customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_keypoints");
		customViewer.viewer->addPointCloudNormals<PointXYZTangent, PointXYZTangent>(scene_keypoints_tangent, scene_keypoints_tangent, 1, 0.005, "scene_keypoints_tangent");
		
		//Draw new Matched model-scene
		for (std::size_t i = 0; i < instances.size(); ++i)
		{
			std::stringstream ss_instance;
			ss_instance << "instance_" << i;
			if (show_FalsePose ? true : hypotheses_mask[i])
			{
				CloudStyle clusterStyle = hypotheses_mask[i] ? style_green : style_red;
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> instance_color_handler(instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
				customViewer.viewer->addPointCloud(instances[i], instance_color_handler, ss_instance.str());
				customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());
			}
		}
		
	}
	cout << "Visualize in " << tt.toc() << " mseconds..." << std::endl;

	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
	cout << "TOTAL 3D MATCHING TIME: " << duration.count() << " mseconds!!" << endl; 
}

double median(cv::Mat channel)
{
	double m = (channel.rows*channel.cols) / 2;
	int bin = 0;
	double med = -1.0;

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cv::Mat hist;
	cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	for (int i = 0; i < histSize && med < 0.0; ++i)
	{
		bin += cvRound(hist.at< float >(i));
		if (bin > m && med < 0.0)
			med = i;
	}

	return med;
}

void DescriptorB2BTL_MEAM::cloudEdgeDetection(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_source, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_extracted, cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_edge)
{
	//Only PointXYZ is suitable
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*cloud_source, *cloud_source_xyz);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extracted_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*cloud_extracted, *cloud_extracted_xyz);

	// Canny Edge detection 2D
	cv::Mat src, src_gray;
	cv::Mat detected_edges;
	
	src = image;
	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
	cv::blur(src_gray, src_gray, cv::Size(3, 3));

	int v = median(src_gray);
	float sigma = 0.33f;
	int lowThreshold = std::max(0, (int)std::floor(v*(1 - sigma)));
	int highThreshold = std::min(255, (int)std::floor(v*(1 + sigma)));
	const int kernel_size = 3;
	Canny(src_gray, detected_edges, 20, 100, kernel_size);

	//Extract edge point indices of source cloud to source_edge_indices
	pcl::PointIndices::Ptr source_edge_indices(new pcl::PointIndices());
	for (int r = 0; r < detected_edges.rows; r++)
		for (int c = 0; c < detected_edges.cols; c++)
			if (detected_edges.at<uint8_t>(r, c) >= 255)
				source_edge_indices->indices.push_back(r*detected_edges.cols + c);
	
	//Get indices of edge point in extracted cloud to extracted_edge_indices
	pcl::PointIndices::Ptr extracted_edge_indices(new pcl::PointIndices());
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_extracted_xyz);

	for (int i = 0; i < static_cast<int>(source_edge_indices->indices.size()); ++i)
	{
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);
		if (!isnan(cloud_source_xyz->at(source_edge_indices->indices.at(i)).x) && kdtree.nearestKSearch(cloud_source_xyz->at(source_edge_indices->indices.at(i)), K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			if (pointNKNSquaredDistance[0] <= 0.001) {
				extracted_edge_indices->indices.push_back(pointIdxNKNSearch[0]);
			}
		}
	}

	//Get edge point of extracted cloud to cloud_extracted_edges
	pcl::ExtractIndices<pcl::PointXYZ> extract_;
	extract_.setInputCloud(cloud_extracted);
	extract_.setIndices(extracted_edge_indices);
	extract_.setNegative(false);
	extract_.filter(*cloud_edge);
}

void DescriptorB2BTL_MEAM::tangentLine(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, pcl::PointCloud<PointXYZTangent>::Ptr& cloud_out)
{	
	pcl::PointCloud<PointXYZTangent>::Ptr cloud_tangent(new pcl::PointCloud<PointXYZTangent>());

	//------------------------------ ROI of Boundary Points -----------------------------
	
	
	//Method #1
	/*
	PointXYZ minPt, maxPt;
	pcl::getMinMax3D(*cloud_in, minPt, maxPt);
	float Hrec = 100 * samp_rad, Wrec = 100 * samp_rad;
	int numSubset = static_cast<int>(std::ceil((maxPt.y - minPt.y) / Hrec) * std::ceil((maxPt.x - minPt.x) / Wrec));

	std::cout << "Edge size: " << cloud_in->size() << " Numsubset: " << numSubset << std::endl;

	std::vector < std::vector<int> > pointSet(numSubset);
	for (int i = 0; i < static_cast<int>(cloud_in->size()); ++i)
	{
		int r = static_cast<int>((cloud_in->at(i).x - minPt.x) / Wrec);
		int c = static_cast<int>((cloud_in->at(i).y - minPt.y) / Hrec);
		pointSet[r*Wrec + c].push_back(i);
	}

	//----------------------------- Tangent Line Calculation -----------------------------
	for (int i = 0; i < static_cast<int>(pointSet.size()); ++i)
	{
		std::vector<int> subPointSet = pointSet[i];
		for (int j = 0; j < static_cast<int>(subPointSet.size()); ++j)
		{
			std::vector<int> score(subPointSet.size());
			for (int k = 0; k < static_cast<int>(subPointSet.size()); ++k)
			{
				if (j == k)
					continue;
				//Count number of points with distance to Vjk < Dthresh
				for (int l = 0; l < static_cast<int>(subPointSet.size()); ++l)
				{
					if (l == j || l == k)
						continue;
					Eigen::Vector3f u, v;
					u << cloud_in->at(subPointSet[k]).x - cloud_in->at(subPointSet[j]).x,
						cloud_in->at(subPointSet[k]).y - cloud_in->at(subPointSet[j]).y,
						cloud_in->at(subPointSet[k]).z - cloud_in->at(subPointSet[j]).z;
					v << cloud_in->at(subPointSet[l]).x - cloud_in->at(subPointSet[j]).x,
						cloud_in->at(subPointSet[l]).y - cloud_in->at(subPointSet[j]).y,
						cloud_in->at(subPointSet[l]).z - cloud_in->at(subPointSet[j]).z;
					float d = (v.cross(u)).norm() / u.norm();
					if (d < Dthresh)
						score[k]++;
				}
			}
			int kStar = std::max_element(score.begin(), score.end()) - score.begin();
			if (score[kStar] > Sthresh * static_cast<int>(subPointSet.size()))
			{
				Eigen::Vector3f t_Pj;
				t_Pj << cloud_in->at(subPointSet[kStar]).x - cloud_in->at(subPointSet[j]).x,
					cloud_in->at(subPointSet[kStar]).y - cloud_in->at(subPointSet[j]).y,
					cloud_in->at(subPointSet[kStar]).z - cloud_in->at(subPointSet[j]).z;
				t_Pj = t_Pj / t_Pj.norm();
				PointXYZTangent Pj;
				Pj.x = cloud_in->at(subPointSet[j]).x;
				Pj.y = cloud_in->at(subPointSet[j]).y;
				Pj.z = cloud_in->at(subPointSet[j]).z;
				Pj.normal_x = t_Pj(0);
				Pj.normal_y = t_Pj(1);
				Pj.normal_z = t_Pj(2);
				cloud_tangent->push_back(Pj);
			}
		}
	}
	*/
	//Method #2
	
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_in);

	unsigned int score = 0;
	for (int index_j = 0; index_j < static_cast<int>(cloud_in->size()); ++index_j)
	{
		int K = 10;
		std::vector<int> subPointSet(K);
		std::vector<float> pointNKNSquaredDistance(K);
		if (kdtree.nearestKSearch(cloud_in->at(index_j), K, subPointSet, pointNKNSquaredDistance) > 0)
		{
			std::vector<int> score(K);

			for (int k = 0; k < K; ++k)
			{
				if (index_j == subPointSet[k])
					continue;
				//Count number of points with distance to Vjk < Dthresh
				for (int l = 0; l < K; ++l)
				{
					if (subPointSet[l] == index_j || subPointSet[l] == subPointSet[k])
						continue;
					Eigen::Vector3f u, v;
					u << cloud_in->at(subPointSet[k]).x - cloud_in->at(index_j).x,
						cloud_in->at(subPointSet[k]).y - cloud_in->at(index_j).y,
						cloud_in->at(subPointSet[k]).z - cloud_in->at(index_j).z;
					v << cloud_in->at(subPointSet[l]).x - cloud_in->at(index_j).x,
						cloud_in->at(subPointSet[l]).y - cloud_in->at(index_j).y,
						cloud_in->at(subPointSet[l]).z - cloud_in->at(index_j).z;
					float d = (v.cross(u)).norm() / u.norm();
					if (d < Dthresh)
						score[k]++;
				}
			}
			int kStar = std::max_element(score.begin(), score.end()) - score.begin();
			if (score[kStar] > Sthresh * static_cast<int>(subPointSet.size()))
			{
				Eigen::Vector3f t_Pj;
				t_Pj << cloud_in->at(subPointSet[kStar]).x - cloud_in->at(index_j).x,
					cloud_in->at(subPointSet[kStar]).y - cloud_in->at(index_j).y,
					cloud_in->at(subPointSet[kStar]).z - cloud_in->at(index_j).z;
				t_Pj = t_Pj / t_Pj.norm();
				PointXYZTangent Pj;
				Pj.x = cloud_in->at(index_j).x;
				Pj.y = cloud_in->at(index_j).y;
				Pj.z = cloud_in->at(index_j).z;
				Pj.normal_x = t_Pj(0);
				Pj.normal_y = t_Pj(1);
				Pj.normal_z = t_Pj(2);
				cloud_tangent->push_back(Pj);
			}
			
		}
	}
	std::cout << "Edge size: " << cloud_tangent->size() << std::endl;
	*cloud_out = *cloud_tangent;
}

