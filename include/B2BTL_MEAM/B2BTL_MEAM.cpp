#include "B2BTL_MEAM.h"

void cloudEdgeDetection(const PointCloudType::ConstPtr& cloud_source, const PointCloudType::ConstPtr& cloud_extracted, Mat image, PointCloudType::Ptr& cloud_edge)
{
	//Only PointXYZ is suitable
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*cloud_source, *cloud_source_xyz);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extracted_xyz(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*cloud_extracted, *cloud_extracted_xyz);

	// Canny Edge detection 2D
	Mat src, src_gray;
	Mat detected_edges;
	int lowThreshold = 3;
	const int max_lowThreshold = 100;
	const int ratio = 3;
	const int kernel_size = 3;
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
	PointType minPt, maxPt;
	pcl::getMinMax3D(*cloud_in, minPt, maxPt);
	float Wrange = maxPt.x - minPt.x, Hrange = maxPt.y - minPt.y;
	int numSubset = 100;
	float Wrec = Wrange / sqrt(numSubset), Hrec = Hrange / sqrt(numSubset);

	std::vector < std::vector<int> > pointSet(numSubset);
	for (int i = 0; i < cloud_in->size(); ++i)
	{
		int r = (int)((cloud_in->at(i).x - minPt.x) / Wrec);
		int c = (int)((cloud_in->at(i).y - minPt.y) / Hrec);
		pointSet[r*Wrec + c].push_back(i);
	}

	for (int subPointSetIndex = 0; subPointSetIndex < pointSet.size(); ++subPointSetIndex)
	{

	}

}