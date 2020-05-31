#include "kinect/kinect.h"
#include "robotArm/robotArm.h"
#include "socket/socket.h"
#include "PPF.h"
#include "SHOT.h"
#include "B2BTL_MEAM.h"

 int main ()
{
	 //Start Kinect
	 Kinect kin;

	 // Choose method DescriptorSHOT, DescriptorPPF or DescriptorB2BTL_MEAM
	 DescriptorB2BTL_MEAM* descr(new DescriptorB2BTL_MEAM(&(kin.images_q)));
	 std::cout << "Descriptor type: " << descr->getType() << endl;
	 
	 descr->prepareModelDescriptor();
	 
	 boost::function<void(const boost::shared_ptr<const PointCloudType >& cloud)> f_cloud =
		 [&descr](const PointCloudType::ConstPtr& cloud) { descr->_3D_Matching(cloud); };
	 kin.registerCallbackFunction(f_cloud);
	 if (!kin.run()) {
		 std::cout << "Error loading Kinect! Exiting... " << std::endl;
		 return -1;
	 }

	 /*
	 //Connect socket
	 Socket soc;
	 int socketConnectResult = soc.createSocketAndConnect();
	 if (socketConnectResult != 0)
	 {
		 cerr << "Can't connect Socket, Err " << socketConnectResult << endl;
		 system("pause");
		 return -1;

	 }
	 */
	

	 while (!descr->customViewer.viewer->wasStopped())
	 {
		 descr->customViewer.viewer->spinOnce(100);
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