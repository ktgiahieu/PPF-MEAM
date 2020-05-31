#include "B2BTL_MEAM.h"

using namespace cv;
using namespace std;
using namespace pcl;

void cloudEdgeDetection(const PointCloudType::ConstPtr& cloud_source, const PointCloudType::ConstPtr& cloud_extracted, Mat image, PointCloudType::Ptr& cloud_edge);
void tangentLine(const PointCloudType::ConstPtr& cloud_in, pcl::PointCloud<PointXYZTangent>::Ptr& cloud_out);

string DescriptorB2BTL_MEAM::getType()
{
	return type;
}

bool DescriptorB2BTL_MEAM::loadModel() {

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

bool DescriptorB2BTL_MEAM::prepareModelDescriptor()
{
	//  Load model cloud
	if (!loadModel())
		return false;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	copyPointCloud(*model, *cloud_xyz);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_HPR(new pcl::PointCloud<pcl::PointXYZ>());
	vector<float> camera_pos{ 0.05, 1, 0.1 };
	HPR(cloud_xyz, camera_pos, 3, cloud_xyz_HPR);
	copyPointCloud(*cloud_xyz_HPR, *model);

	//------------------------------------------------------ Visualization --------------------------------------------------------------------
	customViewer.init();
	pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(scene);
	customViewer.viewer->addPointCloud<PointType>(scene, rgb, "scene", customViewer.v1);
	customViewer.viewer->addPointCloud<PointType>(scene, rgb, "scene_cloud", customViewer.v2);
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene_cloud");

	CloudStyle uniformKeyPointStyle = style_white;

	pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler(scene_keypoints, uniformKeyPointStyle.r, uniformKeyPointStyle.g, uniformKeyPointStyle.b);
	customViewer.viewer->addPointCloud(scene_keypoints, scene_keypoints_color_handler, "scene_keypoints", customViewer.v2);
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

	pcl::transformPointCloud(*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-0.3, 0, 0.9), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler(off_scene_model_keypoints, uniformKeyPointStyle.r, uniformKeyPointStyle.g, uniformKeyPointStyle.b);
	customViewer.viewer->addPointCloud(off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints", customViewer.v2);
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");

	customViewer.viewer->addPointCloudNormals<PointXYZTangent, PointXYZTangent>(scene_keypoints_tangent, scene_keypoints_tangent, 5, 0.005, "scene_keypoints_tangent", customViewer.v2);


	CloudStyle modelPointStyle = style_cyan;

	pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-0.3, 0, 0.9), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler(off_scene_model, modelPointStyle.r, modelPointStyle.g, modelPointStyle.b);
	customViewer.viewer->addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model", customViewer.v2);
	customViewer.viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model");

	customViewer.viewer->addPointCloudNormals<PointXYZTangent, PointXYZTangent>(off_scene_model_keypoints_tangent, off_scene_model_keypoints_tangent, 5, 0.005, "off_scene_model_keypoints_tangent", customViewer.v2);


	return true;
}

void DescriptorB2BTL_MEAM::_3D_Matching(const PointCloudType::ConstPtr &cloud)
{
	mtx.lock();
	tt.tic();

	// --------------------------------------------  Preprocess--------------------------------------------------------------
	*captured_scene = *cloud;
	std::vector<float> filter_limits = { 0.0, 1.2, -0.15, 0.15, -0.12, 0.12 };
	passthrough(captured_scene, filter_limits, passthroughed_scene); // Get limited range point cloud
	sacsegmentation_extindices(passthroughed_scene, 0.008, segmented_scene); // RANSAC Segmentation and remove biggest plane (table)
	statisticalOutlinerRemoval(segmented_scene, 50, statistical_filtered_scene);// 50 k-neighbors noise removal
	*scene = *statistical_filtered_scene; // get the preprocessed scene

	//Check resolution
	float resolution_scene = static_cast<float> (computeCloudResolution(scene));
	cout << "Resolution scene:" << resolution_scene << std::endl;

	// --------------------------------------------  B2B_TL --------------------------------------------------------------
	cloudEdgeDetection(captured_scene, scene, images_q->back(), scene_keypoints);
	images_q->pop();

	tangentLine(scene_keypoints, scene_keypoints_tangent);


	// -------------------------------------------------------- Visualization --------------------------------------------------------

	if (!customViewer.viewer->wasStopped()) {
		customViewer.viewer->updatePointCloud(scene, "scene");
		customViewer.viewer->updatePointCloud(scene, "scene_cloud");
		customViewer.viewer->updatePointCloud(scene_keypoints, "scene_keypoints");
		customViewer.viewer->removePointCloud("scene_keypoints_tangent");
		customViewer.viewer->addPointCloudNormals<PointXYZTangent, PointXYZTangent>(scene_keypoints_tangent, scene_keypoints_tangent, 1, 0.005, "scene_keypoints_tangent", customViewer.v2);
		//customViewer.ready = false;
	}

	cout << "Thoi gian xu li: " << tt.toc() << " ms" << endl;

	// ------------------------ Clear memory --------------------------------
	captured_scene->clear();
	passthroughed_scene->clear();
	voxelgrid_filtered_scene->clear();
	segmented_scene->clear();
	statistical_filtered_scene->clear();
	scene->clear();
	scene_keypoints->clear();
	scene_keypoints_tangent->clear();

	mtx.unlock();
}

void cloudEdgeDetection(const PointCloudType::ConstPtr& cloud_source, const PointCloudType::ConstPtr& cloud_extracted, Mat image, PointCloudType::Ptr& cloud_edge)
{
	//Params
	int lowThreshold = 5;
	const int ratio = 3;
	const int kernel_size = 3;

	//Only PointXYZ is suitable
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*cloud_source, *cloud_source_xyz);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extracted_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*cloud_extracted, *cloud_extracted_xyz);

	// Canny Edge detection 2D
	Mat src, src_gray;
	Mat detected_edges;
	
	src = image;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	
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

	for (int i = 0; i < source_edge_indices->indices.size(); ++i)
	{
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);
		if (!isnan(cloud_source_xyz->at(source_edge_indices->indices.at(i)).x) && kdtree.nearestKSearch(cloud_source_xyz->at(source_edge_indices->indices.at(i)), K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			if (pointNKNSquaredDistance[0] <= 0) {
				extracted_edge_indices->indices.push_back(pointIdxNKNSearch[0]);
			}
		}
	}

	//Get edge point of extracted cloud to cloud_extracted_edges
	pcl::ExtractIndices<PointType> extract_;
	extract_.setInputCloud(cloud_extracted);
	extract_.setIndices(extracted_edge_indices);
	extract_.setNegative(false);
	extract_.filter(*cloud_edge);
}

void tangentLine(const PointCloudType::ConstPtr& cloud_in, pcl::PointCloud<PointXYZTangent>::Ptr& cloud_out)
{	
	//Params
	int numSubset = 100;
	double Dthresh = 0.003; //3mm
	float Sthresh = 0.18;


	pcl::PointCloud<PointXYZTangent>::Ptr cloud_tangent(new pcl::PointCloud<PointXYZTangent>());

	//------------------------------ ROI of Boundary Points -----------------------------
	PointType minPt, maxPt;
	pcl::getMinMax3D(*cloud_in, minPt, maxPt);
	float Wrange = maxPt.x - minPt.x, Hrange = maxPt.y - minPt.y;
	float Wrec = Wrange / sqrt(numSubset), Hrec = Hrange / sqrt(numSubset);

	std::vector < std::vector<int> > pointSet(numSubset);
	for (int i = 0; i < cloud_in->size(); ++i)
	{
		int r = (int)((cloud_in->at(i).x - minPt.x) / Wrec);
		int c = (int)((cloud_in->at(i).y - minPt.y) / Hrec);
		pointSet[r*Wrec + c].push_back(i);
	}

	//----------------------------- Tangent Line Calculation -----------------------------
	for (int i = 0; i < pointSet.size(); ++i)
	{
		std::vector<int> subPointSet = pointSet[i];
		for (int j = 0; j < subPointSet.size(); ++j)
		{
			std::vector<int> score(subPointSet.size());
			for (int k = 0; k < subPointSet.size(); ++k)
			{
				if (j == k)
					continue;
				//Count number of points with distance to Vjk < Dthresh
				for (int l = 0; l < subPointSet.size(); ++l)
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
			if (score[kStar] > Sthresh * subPointSet.size())
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
	*cloud_out = *cloud_tangent;
}


