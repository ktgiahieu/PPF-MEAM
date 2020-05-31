#include "kinect.h"

using namespace cv;
using namespace pcl;
using namespace std;

void Kinect::initIntrinsics()
{
	interface->setRGBCameraIntrinsics(rgb_fx, rgb_fy, rgb_cx, rgb_cy);
	interface->setDepthCameraIntrinsics(depth_fx, depth_fy, depth_cx, depth_cy);
}

void Kinect::registerCallbackFunction(boost::function<void(const boost::shared_ptr<const PointCloudType>& cloud)> lambda_f)
{
	f_cloud = lambda_f;
	f_image = [this](const boost::shared_ptr<pcl::io::Image>& image)
	{
		cv::Mat im = cv::Mat(image->getHeight(), image->getWidth(), CV_8UC3);
		image->fillRGB(im.cols, im.rows, im.data, im.step);
		cv::cvtColor(im, im, CV_RGB2BGR);
		if (images_q.size() < 10)
			images_q.push(im);
	};;
}

bool Kinect::run() {
	initIntrinsics();
	interface->registerCallback(f_cloud);
	interface->registerCallback(f_image);
	interface->start();

	return true;
}

void Kinect::stop() {
	interface->stop();
}
