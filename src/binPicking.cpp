#include "kinect/kinect.h"
#include "robotArm/robotArm.h"
#include "socket/socket.h"

 int main ()
{
	 //pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
	 /*
	 //Connect socket
	 Socket soc;
	 int socketConnectResult = soc.createSocketAndConnect();
	 if (socketConnectResult != 0)
	 {
		 cerr << "Can't connect Socket, Err " << socketConnectResult << endl;
		 system("pause");
		 return -1;
	 }*/

	 //Start Kinect
	 Kinect kin;
	 if (!kin.run()) {
		 std::cout << "Error loading Kinect! Exiting... " << std::endl;
		 return -1;
	 }

	 while (kin.isRunning())
	 {
		 kin.customViewer.viewer->spinOnce(100);
		 boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		 //kin.customViewer.ready = true;
		 //PointCloudType::Ptr result_cloud = PointCloudType::Ptr(new PointCloudType());
		 //kin.getResultCloudTo(result_cloud);
	 }

	 kin.stop();
	 //Close socket
	 //soc.closeAllSockets();
	 return 0;

}