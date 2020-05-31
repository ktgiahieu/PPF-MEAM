#pragma once
#include "pclFunction.h"

using namespace std;
using namespace pcl;

void CustomVisualizer::init() {
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->addText("Scene", 10, 10, "v1 text", v1);
	//viewer->addCoordinateSystem(0.3, v1);

	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
	viewer->addText("Correspondence Grouping", 10, 10, "v2 text", v2);

	//viewer->addCoordinateSystem(0.3, v2);

	viewer->initCameraParameters();
}

/**
* @brief The PassThrough filter is used to identify and/or eliminate points
within a specific range of X, Y and Z values.
* @param cloud_in - input cloud
* @param cloud_out - cloud after the application of the filter - cloud after the application of the filter
* @return void
*/

void passthrough(const PointCloudType::ConstPtr& cloud_in, vector<float> limits, PointCloudType::Ptr& cloud_out)
{
	pcl::PassThrough<PointType> pt;
	
	PointCloudType::Ptr cloud_ptz_ptr(new PointCloudType);
	pt.setInputCloud(cloud_in);
	pt.setFilterFieldName("z");
	//pt.setFilterLimits(0.0, 0.85);//0.81
	pt.setFilterLimits(limits[0], limits[1]);//0.81
	pt.filter(*cloud_ptz_ptr);

	PointCloudType::Ptr cloud_ptx_ptr(new PointCloudType);
	pt.setInputCloud(cloud_ptz_ptr);
	pt.setFilterFieldName("x");
	//pt.setFilterLimits(-0.3, 0.3);//0.37
	pt.setFilterLimits(limits[2], limits[3]);//0.81
	pt.filter(*cloud_ptx_ptr);

	pt.setInputCloud(cloud_ptx_ptr);
	pt.setFilterFieldName("y");
	//pt.setFilterLimits(-0.25, 0.25);//0.2
	pt.setFilterLimits(limits[4], limits[5]);//0.81
	pt.filter(*cloud_out);
}

/*
* @brief The Statistical Outliner Remove filter is used to remove noisy measurements, e.g. outliers, from a point cloud dataset
  using statistical analysis techniques.
* @param cloud_in - input cloud
* @param cloud_out - cloud after the application of the filter
* @return void
*/
void statisticalOutlinerRemoval(const PointCloudType::Ptr& cloud_in, int numNeighbors, PointCloudType::Ptr& cloud_out) {
	// Create the filtering object
	pcl::StatisticalOutlierRemoval<PointType> sor;
	sor.setInputCloud(cloud_in);
	sor.setMeanK(numNeighbors);
	sor.setStddevMulThresh(1.0);
	sor.filter(*cloud_out);
}

/*
* @brief The VoxelGrid filter is used to simplify the cloud, by wrapping the point cloud
  with a three-dimensional grid and reducing the number of points to the center points within each bloc of the grid.
* @param cloud_in - input cloud
* @param cloud_out - cloud after the application of the filter
* @return void
*/
void voxelgrid(const PointCloudType::Ptr& cloud_in, float size_leaf, PointCloudType::Ptr& cloud_out)
{
	// cout << "So diem truoc khi loc voxel " << cloud_in->points.size() << endl;
	pcl::VoxelGrid<PointType> vg;
	vg.setInputCloud(cloud_in);
	vg.setLeafSize(size_leaf, size_leaf, size_leaf); // 3mm
	vg.setDownsampleAllData(true);
	vg.filter(*cloud_out);
	//  cout << "So diem sau khi loc voxel " << cloud_out->points.size() << endl;
}

/*
* @brief The VoxelGrid filter is used to simplify the cloud, by wrapping the point cloud
  with a three-dimensional grid and reducing the number of points to the center points within each bloc of the grid.
* @param cloud_in - input cloud
* @param cloud_out - cloud after the application of the filter
* @return void
*/
void uniformsampling(const PointCloudType::Ptr& cloud_in, float radius, PointCloudType::Ptr& cloud_out)
{
	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud(cloud_in);
	uniform_sampling.setRadiusSearch(radius);
	uniform_sampling.filter(*cloud_out);
}

/*
* @brief The SACSegmentation and the ExtractIndices filters are used to identify and
remove the table from the point cloud leaving only the objects.
* @param cloud_in - input cloud
* @param cloud_out - cloud after the application of the filter - cloud after the application of the filter
* @return void
*/
void sacsegmentation_extindices(const PointCloudType::Ptr& cloud_in,double dist_threshold, PointCloudType::Ptr& cloud_out)
{
	// Identify the table
	pcl::PointIndices::Ptr sacs_inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr sacs_coefficients(new pcl::ModelCoefficients);
	pcl::SACSegmentation<PointType> sacs;
	sacs.setOptimizeCoefficients(true);
	sacs.setModelType(pcl::SACMODEL_PLANE);
	sacs.setMethodType(pcl::SAC_RANSAC);
	sacs.setMaxIterations(900);//900
	sacs.setDistanceThreshold(dist_threshold);//16mm
	sacs.setInputCloud(cloud_in);
	sacs.segment(*sacs_inliers, *sacs_coefficients);
	// Remove the table
	pcl::ExtractIndices<PointType> ei;
	ei.setInputCloud(cloud_in);
	ei.setIndices(sacs_inliers);
	ei.setNegative(true);
	ei.filter(*cloud_out);
}

/*
* @brief The RadiusOutlierRemoval filter is used to remove isolated point according to
the minimum number of neighbors desired.
* @param cloud_in - input cloud
* @param cloud_out - cloud after the application of the filter - cloud after the application of the filter
* @return void
*/
void radiusoutlierremoval(const PointCloudType::Ptr& cloud_in, float radius, uint16_t min_neighbor, PointCloudType::Ptr& cloud_out)
{
	// Remove isolated points
	pcl::RadiusOutlierRemoval<PointType> ror;
	ror.setInputCloud(cloud_in);
	ror.setRadiusSearch(radius);    //2cm        
	ror.setMinNeighborsInRadius(min_neighbor);   //150   
	ror.filter(*cloud_out);
}

/*
* @Tinh phap tuyen dam may diem
* @ cloud_in - cloud input PointType
* @ cloud_out - cloud output XYZRGBNormal
*/
void normal(const PointCloudType::Ptr& cloud_in, int k, float r, char mode, PointCloudNormalType::Ptr& normal_out)
{
	pcl::NormalEstimationOMP<PointType, pcl::Normal> ne;
	pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
	// Calculate all the normals of the entire surface
	ne.setInputCloud(cloud_in);
	ne.setNumberOfThreads(8);
	ne.setSearchMethod(tree);
	if (mode == 'K')
	{
		ne.setKSearch(k);//50
		ne.compute(*normal_out);
	}
	else if (mode == 'R')
	{
		ne.setRadiusSearch(r);//2cm
		ne.compute(*normal_out);
	}
}

double computeCloudResolution(const PointCloudType::ConstPtr &cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<PointType> tree;
	tree.setInputCloud(cloud);

	for (std::size_t i = 0; i < cloud->size(); ++i)
	{
		if (!std::isfinite((*cloud)[i].x))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}
