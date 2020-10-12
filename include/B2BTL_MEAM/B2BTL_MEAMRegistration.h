#ifndef B2BTL_MEAM_REGISTRATION
#define B2BTL_MEAM_REGISTRATION

#include "dataType.h"

#include <pcl/registration/boost.h>
#include <pcl/registration/registration.h>
#include <pcl/features/ppf.h>
#include <algorithm>
#include <pcl/common/transforms.h>
#include <pcl/features/pfh.h>
#include <pcl/registration/icp.h>

#include <unordered_map>

namespace pcl
{
	bool
		computeB2BTL_MEAMPairFeature(const Eigen::Vector4f &p1, const Eigen::Vector4f &n1,
			const Eigen::Vector4f &p2, const Eigen::Vector4f &n2,
			float &f1, float &f2, float &f3, float &f4);

	class B2BTL_MEAMEstimation : public FeatureFromNormals<PointXYZTangent, PointXYZTangent, PPFSignature>
	{
	public:
		typedef boost::shared_ptr<PPFEstimation<PointXYZTangent, PointXYZTangent, PPFSignature> > Ptr;
		typedef boost::shared_ptr<const PPFEstimation<PointXYZTangent, PointXYZTangent, PPFSignature> > ConstPtr;
		using PCLBase<PointXYZTangent>::indices_;
		using Feature<PointXYZTangent, PPFSignature>::input_;
		using Feature<PointXYZTangent, PPFSignature>::feature_name_;
		using Feature<PointXYZTangent, PPFSignature>::getClassName;
		using FeatureFromNormals<PointXYZTangent, PointXYZTangent, PPFSignature>::normals_;

		typedef pcl::PointCloud<PPFSignature> PointCloudOut;

		/** \brief Empty Constructor. */
		B2BTL_MEAMEstimation();


	private:
		/** \brief The method called for actually doing the computations
		  * \param[out] output the resulting point cloud (which should be of type pcl::PPFSignature);
		  * its size is the size of the input cloud, squared (i.e., one point for each pair in
		  * the input cloud);
		  */
		void
			computeFeature(PointCloudOut &output);
		
	};

	class B2BTL_MEAMHashMapSearch
	{
	public:
		/** \brief Data structure to hold the information for the key in the feature hash map of the
		  * B2BTL_MEAMHashMapSearch class
		  * \note It uses multiple pair levels in order to enable the usage of the boost::hash function
		  * which has the std::pair implementation (i.e., does not require a custom hash function)
		  */
		struct HashKeyStruct : public std::pair <int, std::pair <int, std::pair <int, int> > >
		{
			HashKeyStruct () = default;
			HashKeyStruct(int a, int b, int c, int d)
			{
				this->first = a;
				this->second.first = b;
				this->second.second.first = c;
				this->second.second.second = d;
			}
			std::size_t operator()(const HashKeyStruct& s) const noexcept
			{
				const std::size_t h1 = std::hash<int>{} (s.first);
				const std::size_t h2 = std::hash<int>{} (s.second.first);
				const std::size_t h3 = std::hash<int>{} (s.second.second.first);
				const std::size_t h4 = std::hash<int>{} (s.second.second.second);
				return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
			}
		};

		using FeatureHashMapType = std::unordered_multimap<HashKeyStruct, std::pair<std::size_t, std::size_t>, HashKeyStruct>;
      using FeatureHashMapTypePtr = shared_ptr<FeatureHashMapType>;
      using Ptr = shared_ptr<B2BTL_MEAMHashMapSearch>;
      using ConstPtr = shared_ptr<const B2BTL_MEAMHashMapSearch>;

	  using EncodedHashMapType = std::unordered_multimap<HashKeyStruct, std::vector<size_t>, HashKeyStruct>;
	  using EncodedHashMapTypePtr = std::shared_ptr<EncodedHashMapType>;

		//typedef std::unordered_multimap<HashKeyStruct, std::pair<size_t, size_t> > FeatureHashMapType;
		//typedef std::shared_ptr<FeatureHashMapType> FeatureHashMapTypePtr;
		//typedef std::shared_ptr<B2BTL_MEAMHashMapSearch> Ptr;

		//typedef std::unordered_multimap<HashKeyStruct, std::vector<size_t> > EncodedHashMapType;
		//typedef std::shared_ptr<EncodedHashMapType> EncodedHashMapTypePtr;


		/** \brief Constructor for the B2BTL_MEAMHashMapSearch class which sets the two step parameters for the enclosed data structure
		 * \param angle_discretization_step the step value between each bin of the hash map for the angular values
		 * \param distance_discretization_step the step value between each bin of the hash map for the distance values
		 */
		B2BTL_MEAMHashMapSearch(float angle_discretization_step = 12.0f / 180.0f * static_cast<float> (M_PI),
			float distance_discretization_step = 0.01f)
			: alpha_m_EAM()
			, feature_hash_map_(new FeatureHashMapType)
			, encoded_hash_map_(new EncodedHashMapType)
			, internals_initialized_(false)
			, angle_discretization_step_(angle_discretization_step)
			, distance_discretization_step_(distance_discretization_step)
			, max_dist_(-1.0f)
			, Lvoxel(0.01f)
		{
		}

		/** \brief Method that sets the feature cloud to be inserted in the hash map
		 * \param feature_cloud a const smart pointer to the PPFSignature feature cloud
		 */
		void
			setInputFeatureCloud(PointCloud<PPFSignature>::ConstPtr feature_cloud, PointCloud<PointXYZ>::ConstPtr cloud_XYZ);

		/** \brief Function for finding the nearest neighbors for the given feature inside the discretized hash map
		 * \param f1 The 1st value describing the query PPFSignature feature
		 * \param f2 The 2nd value describing the query PPFSignature feature
		 * \param f3 The 3rd value describing the query PPFSignature feature
		 * \param f4 The 4th value describing the query PPFSignature feature
		 * \param indices a vector of pair indices representing the feature pairs that have been found in the bin
		 * corresponding to the query feature
		 */
		void
			nearestNeighborSearch(float &f1, float &f2, float &f3, float &f4,
				std::vector<std::pair<size_t, size_t> > &indices);

		/** \brief Convenience method for returning a copy of the class instance as a boost::shared_ptr */
		Ptr
			makeShared() { return Ptr(new B2BTL_MEAMHashMapSearch(*this)); }

		/** \brief Returns the angle discretization step parameter (the step value between each bin of the hash map for the angular values) */
		inline float
			getAngleDiscretizationStep() { return angle_discretization_step_; }

		/** \brief Returns the distance discretization step parameter (the step value between each bin of the hash map for the distance values) */
		inline float
			getDistanceDiscretizationStep() { return distance_discretization_step_; }

		/** \brief Returns the maximum distance found between any feature pair in the given input feature cloud */
		inline float
			getModelDiameter() { return max_dist_; }

		std::vector<std::vector <std::vector <float> > > alpha_m_EAM;
		std::vector <size_t> EAM_end_indices{ 0 };
		float Lvoxel;
	private:
		FeatureHashMapTypePtr feature_hash_map_;
		EncodedHashMapTypePtr encoded_hash_map_;
		bool internals_initialized_;

		float angle_discretization_step_, distance_discretization_step_;
		float max_dist_;
	};

	/** \brief Class that registers two point clouds based on their sets of PPFSignatures.
	 * Please refer to the following publication for more details:
	 *    B. Drost, M. Ulrich, N. Navab, S. Ilic
	 *    Model Globally, Match Locally: Efficient and Robust 3D Object Recognition
	 *    2010 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
	 *    13-18 June 2010, San Francisco, CA
	 *
	 * \note This class works in tandem with the PPFEstimation class
	 *
	 * \author Alexandru-Eugen Ichim
	 */
	class B2BTL_MEAMRegistration : public Registration<PointXYZTangent, PointXYZTangent>
	{
	public:
		/** \brief Structure for storing a pose (represented as an Eigen::Affine3f) and an integer for counting votes
		  * \note initially used std::pair<Eigen::Affine3f, unsigned int>, but it proved problematic
		  * because of the Eigen structures alignment problems - std::pair does not have a custom allocator
		  */
		struct PoseWithVotes
		{
			PoseWithVotes(Eigen::Affine3f &a_pose, unsigned int &a_votes, int &a_viewpoint)
				: pose(a_pose),
				votes(a_votes),
				viewpoint(a_viewpoint)
			{}

			Eigen::Affine3f pose;
			unsigned int votes;
			int viewpoint;
		};
		typedef std::vector<PoseWithVotes, Eigen::aligned_allocator<PoseWithVotes> > PoseWithVotesList;

		/// input_ is the model cloud
		using Registration<PointXYZTangent, PointXYZTangent>::input_;
		/// target_ is the scene cloud
		using Registration<PointXYZTangent, PointXYZTangent>::target_;
		using Registration<PointXYZTangent, PointXYZTangent>::converged_;
		using Registration<PointXYZTangent, PointXYZTangent>::final_transformation_;
		using Registration<PointXYZTangent, PointXYZTangent>::transformation_;

		typedef pcl::PointCloud<PointXYZTangent> PointCloudSource;
		typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
		typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

		typedef pcl::PointCloud<PointXYZTangent> PointCloudTarget;
		typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
		typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;


		/** \brief Empty constructor that initializes all the parameters of the algorithm with default values */
		B2BTL_MEAMRegistration()
			: Registration<PointXYZTangent, PointXYZTangent>(),
			search_method_(),
			scene_reference_point_sampling_rate_(5),
			clustering_position_diff_threshold_(0.01f),
			clustering_rotation_diff_threshold_(20.0f / 180.0f * static_cast<float> (M_PI)),
			hv_dist_thresh(0.01f),
			Npv(10),
			icp_max_iter_(3),
			icp_corr_distance_(0.05f)
		{}

		/** \brief Method for setting the position difference clustering parameter
		 * \param clustering_position_diff_threshold distance threshold below which two poses are
		 * considered close enough to be in the same cluster (for the clustering phase of the algorithm)
		 */
		inline void
			setPositionClusteringThreshold(float clustering_position_diff_threshold) { clustering_position_diff_threshold_ = clustering_position_diff_threshold; }

		/** \brief Returns the parameter defining the position difference clustering parameter -
		 * distance threshold below which two poses are considered close enough to be in the same cluster
		 * (for the clustering phase of the algorithm)
		 */
		inline float
			getPositionClusteringThreshold() { return clustering_position_diff_threshold_; }

		/** \brief Method for setting the rotation clustering parameter
		 * \param clustering_rotation_diff_threshold rotation difference threshold below which two
		 * poses are considered to be in the same cluster (for the clustering phase of the algorithm)
		 */
		inline void
			setRotationClusteringThreshold(float clustering_rotation_diff_threshold) { clustering_rotation_diff_threshold_ = clustering_rotation_diff_threshold; }

		/** \brief Returns the parameter defining the rotation clustering threshold
		 */
		inline float
			getRotationClusteringThreshold() { return clustering_rotation_diff_threshold_; }

		/** \brief Method for setting the scene reference point sampling rate
		 * \param scene_reference_point_sampling_rate sampling rate for the scene reference point
		 */
		inline void
			setSceneReferencePointSamplingRate(unsigned int scene_reference_point_sampling_rate) { scene_reference_point_sampling_rate_ = scene_reference_point_sampling_rate; }

		/** \brief Returns the parameter for the scene reference point sampling rate of the algorithm */
		inline unsigned int
			getSceneReferencePointSamplingRate() { return scene_reference_point_sampling_rate_; }

		/** \brief Function that sets the search method for the algorithm
		 * \note Right now, the only available method is the one initially proposed by
		 * the authors - by using a hash map with discretized feature vectors
		 * \param search_method smart pointer to the search method to be set
		 */
		inline void
			setSearchMethod(B2BTL_MEAMHashMapSearch::Ptr search_method) { search_method_ = search_method; }

		/** \brief Getter function for the search method of the class */
		inline B2BTL_MEAMHashMapSearch::Ptr
			getSearchMethod() { return search_method_; }

		/** \brief Provide a pointer to the input target (e.g., the point cloud that we want to align the input source to)
		 * \param cloud the input point cloud target
		 */
		void
			setInputTarget(const PointCloudTargetConstPtr &cloud);

		/** \brief Provide a pointer to the input target (e.g., the point cloud that we want to align the input source to)
		 * \param cloud the input point cloud target
		 */
		void
			setListInputSource(const std::vector< pcl::PointCloud<PointXYZTangent>::Ptr>& cloud_list);

		void
			computeFinalPoses(typename pcl::B2BTL_MEAMRegistration::PoseWithVotesList &result);

		inline void
			setHVDistanceThresh(float a_hv_dist_thresh) { hv_dist_thresh = a_hv_dist_thresh; }

		inline void
			setNpv(size_t a_Npv) { Npv = a_Npv; }

		inline void
			setICPMaxIterations(int a_icp_max_iter_) { icp_max_iter_ = a_icp_max_iter_; }

		inline void
			setICPCorrespondenceDistanceThreshold(float a_icp_corr_distance_) { icp_corr_distance_ = a_icp_corr_distance_; }

	private:
		float hv_dist_thresh;
		size_t Npv;
		int icp_max_iter_;
		float icp_corr_distance_;
		/** \brief Method that calculates the transformation between the input_ and target_ point clouds, based on the PPF features */
		void
			computeTransformation(PointCloudSource &output, const Eigen::Matrix4f& guess);

		std::vector<pcl::PointCloud<PointXYZTangent>::Ptr> input_list;

		/** \brief the search method that is going to be used to find matching feature pairs */
		B2BTL_MEAMHashMapSearch::Ptr search_method_;

		/** \brief parameter for the sampling rate of the scene reference points */
		unsigned int scene_reference_point_sampling_rate_;

		/** \brief position and rotation difference thresholds below which two
		  * poses are considered to be in the same cluster (for the clustering phase of the algorithm) */
		float clustering_position_diff_threshold_, clustering_rotation_diff_threshold_;

		/** \brief use a kd-tree with range searches of range max_dist to skip an O(N) pass through the point cloud */
		typename pcl::KdTreeFLANN<PointXYZTangent>::Ptr scene_search_tree_;

		/** \brief static method used for the std::sort function to order two PoseWithVotes
		 * instances by their number of votes*/
		static bool
			poseWithVotesCompareFunction(const PoseWithVotes &a,
				const PoseWithVotes &b);

		/** \brief static method used for the std::sort function to order two pairs <index, votes>
		 * by the number of votes (unsigned integer value) */
		static bool
			clusterVotesCompareFunction(const std::pair<size_t, unsigned int> &a,
				const std::pair<size_t, unsigned int> &b);

		/** \brief Method that clusters a set of given poses by using the clustering thresholds
		 * and their corresponding number of votes (see publication for more details) */
		void
			clusterPoses(PoseWithVotesList &poses,
				PoseWithVotesList &result);

		/** \brief Method that checks whether two poses are close together - based on the clustering threshold parameters
		 * of the class */
		bool
			posesWithinErrorBounds(Eigen::Affine3f &pose1,
				Eigen::Affine3f &pose2);
	};
}

#endif //B2BTL_MEAM_REGISTRATION