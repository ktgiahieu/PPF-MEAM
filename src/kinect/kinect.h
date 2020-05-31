#ifndef KINECT
#define KINECT

#include "dataType.h"

// io
#include <iostream>
#include <queue>
#include <string>
#include <pcl/io/openni2_grabber.h>

//opencv
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


class Kinect
{
private:

	//Intrinsics
	double rgb_fx = 527.360558216633;
	double rgb_fy = 528.3632079059613;
	double rgb_cx = 319.6283688314646;
	double rgb_cy = 263.2500840924527;

	double depth_fx = 585.3081185218954;
	double depth_fy = 587.28826280342;
	double depth_cx = 317.8356267919822;
	double depth_cy = 247.1889948214958;

	
	//Grabber
	pcl::io::OpenNI2Grabber* interface = new pcl::io::OpenNI2Grabber("B00364608306123B");
	
	const int max_images = 10;
	
	//Callbacks
	boost::function<void(const boost::shared_ptr<const PointCloudType >& cloud)> f_cloud;
	boost::function<void(const boost::shared_ptr<pcl::io::Image>& image)> f_image;


public:
	std::queue<cv::Mat> images_q;
	Kinect(){}
	void initIntrinsics();
	void registerCallbackFunction(boost::function<void(const boost::shared_ptr<const PointCloudType>& cloud)> lambda_f);
	bool run();
	void stop();
};

#endif