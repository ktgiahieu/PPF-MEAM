//Author: Khuong Thanh Gia Hieu
//Bach Khoa University - CK16KSCD

#include "B2BTL_MEAM.h"
#include <thread>
#include <mutex>

int main()
{
	//In this project, we use B2BTL - MEAM Descriptor from this paper, please read before continue:
	//Point Pair Feature-Based Pose Estimation with Multiple Edge Appearance Models (PPF-MEAM) for Robotic Bin Picking
    //Diyi Liu, Shogo Arai, Jiaqi Miao, Jun Kinugawa, Zhao Wang and Kazuhiro Kosuge
	//Sensors 2018
	DescriptorB2BTL_MEAM* descr(new DescriptorB2BTL_MEAM());
	std::cout << "Descriptor type: " << descr->getType() << endl;
	
	//Load model
	descr->setModelPath("../data/model/nap/nap.STL");
	
	std::cout << "Done Preparation ... !" << std::endl;

	//Load scene cloud
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> ("../data/scene/nap/scene4.pcd", *cloud) == -1) //* load the file
    {
		PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
		return (-1);
    }
	descr->storeLatestCloud(cloud);

	//Load scene image
	cv::Mat im = cv::imread("../data/scene/nap/scene4.jpg", 1);
	descr->storeLatestImage(im);

	//We MUST process and visualize from different thread, or the program will crash

	// Start processing from different thread
	auto _3D_Matching_Lambda = [&descr]() {
		descr->prepareModelDescriptor();
		descr->_3D_Matching();
	};
	std::thread _3D_Matching_Thread(_3D_Matching_Lambda);
	std::cout << "Processing Thread Started ... !" << std::endl;
	
	// Start visualizing from different thread
	while (!descr->customViewer.viewer->wasStopped()) {
		descr->customViewer.viewer->spinOnce(300);
		std::this_thread::sleep_for(std::chrono::microseconds(300000));
	}
	 
	//Wait for thread to finish before closing the program
	if (_3D_Matching_Thread.joinable())
		_3D_Matching_Thread.join();

	return 0;

}
