#include "B2BTL_MEAMRegistration.h"

bool
pcl::computeB2BTL_MEAMPairFeature(const Eigen::Vector4f &p1, const Eigen::Vector4f &n1,
	const Eigen::Vector4f &p2, const Eigen::Vector4f &n2,
	float &f1, float &f2, float &f3, float &f4)
{
	Eigen::Vector4f delta = p2 - p1;
	delta[3] = 0.0f;
	// f4 = ||delta||
	f4 = delta.norm();

	delta /= f4;

	// f1 = n1 dot delta
	f1 = n1[0] * delta[0] + n1[1] * delta[1] + n1[2] * delta[2];
	// f2 = n2 dot delta
	f2 = n2[0] * delta[0] + n2[1] * delta[1] + n2[2] * delta[2];
	// f3 = n1 dot n2
	f3 = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];

	return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////
pcl::B2BTL_MEAMEstimation::B2BTL_MEAMEstimation()
	: FeatureFromNormals <PointXYZTangent, PointXYZTangent, pcl::PPFSignature>()
{
	feature_name_ = "B2BTL_MEAMEstimation";
	// Slight hack in order to pass the check for the presence of a search method in Feature::initCompute ()
	Feature<PointXYZTangent, PPFSignature>::tree_.reset(new pcl::search::KdTree <PointXYZTangent>());
	Feature<PointXYZTangent, PPFSignature>::search_radius_ = 1.0f;
}


//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::B2BTL_MEAMEstimation::computeFeature(PointCloudOut &output)
{
	// Initialize output container - overwrite the sizes done by Feature::initCompute ()
	output.points.resize(indices_->size() * input_->points.size());
	output.height = 1;
	output.width = static_cast<uint32_t> (output.points.size());
	output.is_dense = true;

	// Compute point pair features for every pair of points in the cloud
	for (size_t index_i = 0; index_i < indices_->size(); ++index_i)
	{
		size_t i = (*indices_)[index_i];
		for (size_t j = 0; j < input_->points.size(); ++j)
		{
			pcl::PPFSignature p;
			if (i != j)
			{
				if (pcl::computeB2BTL_MEAMPairFeature(input_->points[i].getVector4fMap(),
						normals_->points[i].getNormalVector4fMap(),
						input_->points[j].getVector4fMap(),
						normals_->points[j].getNormalVector4fMap(),
						p.f1, p.f2, p.f3, p.f4))
				{
					// Calculate alpha_m angle
					Eigen::Vector3f model_reference_point = input_->points[i].getVector3fMap(),
						model_reference_normal = normals_->points[i].getNormalVector3fMap(),
						model_point = input_->points[j].getVector3fMap();
					float rotation_angle = acosf(model_reference_normal.dot(Eigen::Vector3f::UnitX()));
					bool parallel_to_x = (model_reference_normal.y() == 0.0f && model_reference_normal.z() == 0.0f);
					Eigen::Vector3f rotation_axis = (parallel_to_x) ? (Eigen::Vector3f::UnitY()) : (model_reference_normal.cross(Eigen::Vector3f::UnitX()).normalized());
					Eigen::AngleAxisf rotation_mg(rotation_angle, rotation_axis);
					Eigen::Affine3f transform_mg(Eigen::Translation3f(rotation_mg * ((-1) * model_reference_point)) * rotation_mg);

					Eigen::Vector3f model_point_transformed = transform_mg * model_point;
					float angle = atan2f(-model_point_transformed(2), model_point_transformed(1));
					if (sin(angle) * model_point_transformed(2) < 0.0f)
						angle *= (-1);
					p.alpha_m = -angle;
				}
				else
				{
					PCL_ERROR("[pcl::%s::computeFeature] Computing pair feature vector between points %u and %u went wrong.\n", getClassName().c_str(), i, j);
					p.f1 = p.f2 = p.f3 = p.f4 = p.alpha_m = std::numeric_limits<float>::quiet_NaN();
					output.is_dense = false;
				}
			}
			// Do not calculate the feature for identity pairs (i, i) as they are not used
			// in the following computations
			else
			{
				p.f1 = p.f2 = p.f3 = p.f4 = p.alpha_m = std::numeric_limits<float>::quiet_NaN();
				output.is_dense = false;
			}

			output.points[index_i*input_->points.size() + j] = p;
		}
	}
}



void
pcl::B2BTL_MEAMHashMapSearch::setInputFeatureCloud(PointCloud<PPFSignature>::ConstPtr feature_cloud, PointCloud<PointXYZ>::ConstPtr cloud_XYZ)
{
	// Discretize the feature cloud and insert it in the hash map
	unsigned int n = static_cast<unsigned int> (std::sqrt(static_cast<float> (feature_cloud->points.size())));
	int d1, d2, d3, d4;
	alpha_m_EAM.push_back({});
	std::vector < std::vector <float> >& alpha_m_ = alpha_m_EAM.back();
	alpha_m_.resize(n);
	for (std::size_t i = 0; i < n; ++i)
	{
		std::vector <float> alpha_m_row(n);
		for (std::size_t j = 0; j < n; ++j)
		{
			d1 = static_cast<int> (std::floor(feature_cloud->points[i*n + j].f1 / angle_discretization_step_));
			d2 = static_cast<int> (std::floor(feature_cloud->points[i*n + j].f2 / angle_discretization_step_));
			d3 = static_cast<int> (std::floor(feature_cloud->points[i*n + j].f3 / angle_discretization_step_));
			d4 = static_cast<int> (std::floor(feature_cloud->points[i*n + j].f4 / distance_discretization_step_));
			HashKeyStruct key = HashKeyStruct(d1, d2, d3, d4);

			//Check for duplicate
			std::vector<size_t> encoded_pos{ static_cast<size_t>(cloud_XYZ->at(i).x / Lvoxel), static_cast<size_t>(cloud_XYZ->at(i).y / Lvoxel) ,static_cast<size_t>(cloud_XYZ->at(i).z / Lvoxel) ,static_cast<size_t>(cloud_XYZ->at(j).x / Lvoxel) , static_cast<size_t>(cloud_XYZ->at(j).y / Lvoxel) , static_cast<size_t>(cloud_XYZ->at(j).z / Lvoxel) };
			auto encoded_range = encoded_hash_map_->equal_range(key);
			bool already_added = false; 
			for (; encoded_range.first != encoded_range.second; ++encoded_range.first)
			{
				if (encoded_range.first->second == encoded_pos)
					already_added = true;
			}
			if (already_added)
				continue;
			encoded_hash_map_->insert(std::pair<HashKeyStruct, std::vector <size_t> >(key, encoded_pos));

			feature_hash_map_->insert(std::pair<HashKeyStruct, std::pair<std::size_t, std::size_t> >(key, std::pair<std::size_t, std::size_t>(EAM_end_indices.back() + i, EAM_end_indices.back() + j)));
			alpha_m_row[j] = feature_cloud->points[i*n + j].alpha_m;

			if (max_dist_ < feature_cloud->points[i*n + j].f4)
				max_dist_ = feature_cloud->points[i*n + j].f4;
		}
		alpha_m_[i] = alpha_m_row;
	}

	EAM_end_indices.push_back(n + EAM_end_indices.back());

	internals_initialized_ = true;
}


//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::B2BTL_MEAMHashMapSearch::nearestNeighborSearch(float &f1, float &f2, float &f3, float &f4,
	std::vector<std::pair<std::size_t, std::size_t> > &indices)
{
	if (!internals_initialized_)
	{
		PCL_ERROR("[pcl::B2BTL_MEAMRegistration::nearestNeighborSearch]: input feature cloud has not been set - skipping search!\n");
		return;
	}

	int d1 = static_cast<int> (std::floor(f1 / angle_discretization_step_)),
		d2 = static_cast<int> (std::floor(f2 / angle_discretization_step_)),
		d3 = static_cast<int> (std::floor(f3 / angle_discretization_step_)),
		d4 = static_cast<int> (std::floor(f4 / distance_discretization_step_));

	indices.clear();
	HashKeyStruct key = HashKeyStruct(d1, d2, d3, d4);
	auto map_iterator_pair = feature_hash_map_->equal_range(key);
	for (; map_iterator_pair.first != map_iterator_pair.second; ++map_iterator_pair.first)
		indices.emplace_back(map_iterator_pair.first->second.first,
			map_iterator_pair.first->second.second);
}

void
pcl::B2BTL_MEAMRegistration::setInputTarget(const PointCloudTargetConstPtr &cloud)
{
	Registration<PointXYZTangent, PointXYZTangent>::setInputTarget(cloud);

	scene_search_tree_ = typename pcl::KdTreeFLANN<PointXYZTangent>::Ptr(new pcl::KdTreeFLANN<PointXYZTangent>);
	scene_search_tree_->setInputCloud(target_);
}

void
pcl::B2BTL_MEAMRegistration::setListInputSource(const std::vector< pcl::PointCloud<PointXYZTangent>::Ptr>& cloud_list)
{
	input_list.clear();
	for (int i = 0; i < cloud_list.size(); ++i)
	{
		input_list.push_back(cloud_list[i]);
	}
		
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::B2BTL_MEAMRegistration::computeTransformation(PointCloudSource &output, const Eigen::Matrix4f& guess)
{
	if (!search_method_)
	{
		PCL_ERROR("[pcl::B2BTL_MEAMRegistration::computeTransformation] Search method not set - skipping computeTransformation!\n");
		return;
	}

	if (guess != Eigen::Matrix4f::Identity())
	{
		PCL_ERROR("[pcl::B2BTL_MEAMRegistration::computeTransformation] setting initial transform (guess) not implemented!\n");
	}

	PoseWithVotesList voted_poses;
	std::vector <std::vector <unsigned int> > accumulator_array;
	accumulator_array.resize(search_method_->EAM_end_indices.back());

	size_t aux_size = static_cast<size_t> (floor(2 * M_PI / search_method_->getAngleDiscretizationStep()));
	for (size_t i = 0; i < search_method_->EAM_end_indices.back(); ++i)
	{
		std::vector<unsigned int> aux(aux_size);
		accumulator_array[i] = aux;
	}
	PCL_INFO("Accumulator array size: %u x %u.\n", accumulator_array.size(), accumulator_array.back().size());

	// Consider every <scene_reference_point_sampling_rate>-th point as the reference point => fix s_r
	float f1, f2, f3, f4;
	for (size_t scene_reference_index = 0; scene_reference_index < target_->points.size(); scene_reference_index += scene_reference_point_sampling_rate_)
	{
		Eigen::Vector3f scene_reference_point = target_->points[scene_reference_index].getVector3fMap(),
			scene_reference_normal = target_->points[scene_reference_index].getNormalVector3fMap();

		float rotation_angle_sg = acosf(scene_reference_normal.dot(Eigen::Vector3f::UnitX()));
		bool parallel_to_x_sg = (scene_reference_normal.y() == 0.0f && scene_reference_normal.z() == 0.0f);
		Eigen::Vector3f rotation_axis_sg = (parallel_to_x_sg) ? (Eigen::Vector3f::UnitY()) : (scene_reference_normal.cross(Eigen::Vector3f::UnitX()).normalized());
		Eigen::AngleAxisf rotation_sg(rotation_angle_sg, rotation_axis_sg);
		Eigen::Affine3f transform_sg(Eigen::Translation3f(rotation_sg * ((-1) * scene_reference_point)) * rotation_sg);

		// For every other point in the scene => now have pair (s_r, s_i) fixed
		std::vector<int> indices;
		std::vector<float> distances;
		scene_search_tree_->radiusSearch(target_->points[scene_reference_index],
			search_method_->getModelDiameter() / 2,
			indices,
			distances);
		for (size_t i = 0; i < indices.size(); ++i)
			//    for(size_t i = 0; i < target_->points.size (); ++i)
		{
			//size_t scene_point_index = i;
			size_t scene_point_index = indices[i];
			if (scene_reference_index != scene_point_index)
			{
				if (/*pcl::computePPFPairFeature*/pcl::computePairFeatures(target_->points[scene_reference_index].getVector4fMap(),
					target_->points[scene_reference_index].getNormalVector4fMap(),
					target_->points[scene_point_index].getVector4fMap(),
					target_->points[scene_point_index].getNormalVector4fMap(),
					f1, f2, f3, f4))
				{
					std::vector<std::pair<size_t, size_t> > nearest_indices;
					search_method_->nearestNeighborSearch(f1, f2, f3, f4, nearest_indices);

					// Compute alpha_s angle
					Eigen::Vector3f scene_point = target_->points[scene_point_index].getVector3fMap();

					Eigen::Vector3f scene_point_transformed = transform_sg * scene_point;
					float alpha_s = atan2f(-scene_point_transformed(2), scene_point_transformed(1));
					if (sin(alpha_s) * scene_point_transformed(2) < 0.0f)
						alpha_s *= (-1);
					alpha_s *= (-1);

					// Go through point pairs in the model with the same discretized feature
					for (std::vector<std::pair<size_t, size_t> >::iterator v_it = nearest_indices.begin(); v_it != nearest_indices.end(); ++v_it)
					{
						int viewpoint = 0;
						for (; viewpoint < search_method_->EAM_end_indices.size(); ++viewpoint)
						{
							if (v_it->first < search_method_->EAM_end_indices[viewpoint + 1])
								break;
						}
						
						size_t model_reference_index = v_it->first - search_method_->EAM_end_indices[viewpoint],
							model_point_index = v_it->second - search_method_->EAM_end_indices[viewpoint];
						// Calculate angle alpha = alpha_m - alpha_s
						float alpha = search_method_->alpha_m_EAM[viewpoint][model_reference_index][model_point_index] - alpha_s;

						if (alpha < -M_PI)
							alpha += (M_PI * 2.0);
						if (alpha > M_PI)
							alpha -= (M_PI * 2.0);

						unsigned int alpha_discretized = static_cast<unsigned int> (floor(alpha + M_PI) / search_method_->getAngleDiscretizationStep());
						
						accumulator_array[v_it->first][alpha_discretized] ++;
					}
				}
				else PCL_ERROR("[pcl::B2BTL_MEAMRegistration::computeTransformation] Computing pair feature vector between points %u and %u went wrong.\n", scene_reference_index, scene_point_index);
			}
		}

		size_t max_votes_i = 0, max_votes_j = 0;
		unsigned int max_votes = 0;

		for (size_t i = 0; i < accumulator_array.size(); ++i)
			for (size_t j = 0; j < accumulator_array.back().size(); ++j)
			{
				if (accumulator_array[i][j] > max_votes)
				{
					max_votes = accumulator_array[i][j];
					max_votes_i = i;
					max_votes_j = j;
				}
				// Reset accumulator_array for the next set of iterations with a new scene reference point
				accumulator_array[i][j] = 0;
			}

		int viewpoint = 0;
		for (; viewpoint < search_method_->EAM_end_indices.size(); ++viewpoint)
		{
			if (max_votes_i < search_method_->EAM_end_indices[viewpoint + 1])
				break;
		}


		Eigen::Vector3f model_reference_point = input_list[viewpoint]->points[max_votes_i - search_method_->EAM_end_indices[viewpoint]].getVector3fMap(),
						model_reference_normal = input_list[viewpoint]->points[max_votes_i - search_method_->EAM_end_indices[viewpoint]].getNormalVector3fMap();

		float rotation_angle_mg = acosf(model_reference_normal.dot(Eigen::Vector3f::UnitX()));
		bool parallel_to_x_mg = (model_reference_normal.y() == 0.0f && model_reference_normal.z() == 0.0f);
		Eigen::Vector3f rotation_axis_mg = (parallel_to_x_mg) ? (Eigen::Vector3f::UnitY()) : (model_reference_normal.cross(Eigen::Vector3f::UnitX()).normalized());
		Eigen::AngleAxisf rotation_mg(rotation_angle_mg, rotation_axis_mg);
		Eigen::Affine3f transform_mg(Eigen::Translation3f(rotation_mg * ((-1) * model_reference_point)) * rotation_mg);
		Eigen::Affine3f max_transform =
			transform_sg.inverse() *
			Eigen::AngleAxisf((static_cast<float> (max_votes_j) - floorf(static_cast<float> (M_PI) / search_method_->getAngleDiscretizationStep())) * search_method_->getAngleDiscretizationStep(), Eigen::Vector3f::UnitX()) *
			transform_mg;

		voted_poses.push_back(PoseWithVotes(max_transform, max_votes, viewpoint));
	}
	PCL_DEBUG("Done with the Hough Transform ...\n");

	// Cluster poses for filtering out outliers and obtaining more precise results
	PoseWithVotesList results;
	clusterPoses(voted_poses, results);

	transformation_ = final_transformation_ = results.front().pose.matrix();
	converged_ = true;
}


//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::B2BTL_MEAMRegistration::clusterPoses(typename pcl::B2BTL_MEAMRegistration::PoseWithVotesList &poses,
	typename pcl::B2BTL_MEAMRegistration::PoseWithVotesList &result)
{
	PCL_INFO("Clustering poses ...\n");
	// Start off by sorting the poses by the number of votes
	sort(poses.begin(), poses.end(), poseWithVotesCompareFunction);
	for (size_t pose_index = 0; pose_index < poses.size(); ++pose_index)
	{
		if (poses[pose_index].votes < 0.2*poses[0].votes || pose_index > 50)
		{
			poses.erase(poses.begin() + pose_index, poses.end());
			break;
		}
	}

	PoseWithVotesList poses_score;
	for (size_t poses_i = 0; poses_i < poses.size() && poses[poses_i].votes > 0.3 * poses[0].votes; ++poses_i)
	{
		pcl::PointCloud<PointXYZTangent>::Ptr instance(new pcl::PointCloud<PointXYZTangent>());
		copyPointCloud(*(input_list[poses[poses_i].viewpoint]), *instance);
		pcl::transformPointCloud(*instance, *instance, poses[poses_i].pose);

		pcl::KdTreeFLANN<PointXYZTangent> kdtree;
		kdtree.setInputCloud(target_);

		unsigned int score = 0;
		for (int i = 0; i < static_cast<int>(instance->size()); ++i)
		{
			int K = 1;
			std::vector<int> pointIdxNKNSearch(K);
			std::vector<float> pointNKNSquaredDistance(K);
			if (kdtree.nearestKSearch(instance->at(i), K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
			{
				if (pointNKNSquaredDistance[0] < hv_dist_thresh)
					score++;
			}
		}
		poses_score.push_back(poses[poses_i]);
		poses_score[poses_i].votes = score;
	}

	sort(poses_score.begin(), poses_score.end(), poseWithVotesCompareFunction);


	PoseWithVotesList poses_high_score_vp;
	std::vector<int> numPosesFromViewpoint(input_list.size());
	for (size_t poses_i = 0; poses_i < poses_score.size(); ++poses_i)
	{
		if (numPosesFromViewpoint[poses_score[poses_i].viewpoint] < Npv)
		{
			numPosesFromViewpoint[poses_score[poses_i].viewpoint]++;
			poses_high_score_vp.push_back(poses_score[poses_i]);
		}
	}



	PoseWithVotesList clusters;
	for (size_t poses_i = 0; poses_i < poses_high_score_vp.size(); ++poses_i)
	{
		bool found_cluster = false;
		for (size_t clusters_i = 0; clusters_i < clusters.size(); ++clusters_i)
		{
			if (posesWithinErrorBounds(poses_high_score_vp[poses_i].pose, clusters[clusters_i].pose))
			{
				found_cluster = true;
				if (poses_high_score_vp[poses_i].votes > clusters[clusters_i].votes)
					clusters[clusters_i] = poses_high_score_vp[poses_i];
				break;
			}
		}

		if (found_cluster == false)
		{
			// Create a new cluster with the current pose
			clusters.push_back(poses_high_score_vp[poses_i]);
		}
	}

	sort(clusters.begin(), clusters.end(), poseWithVotesCompareFunction);

	result.clear();
	for (size_t clusters_i = 0; clusters_i < clusters.size(); ++clusters_i)
	{
		PCL_INFO("Winning cluster has #scores: %d.\n", clusters[clusters_i].votes);

		result.push_back(clusters[clusters_i]);
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::B2BTL_MEAMRegistration::posesWithinErrorBounds(Eigen::Affine3f &pose1,
	Eigen::Affine3f &pose2)
{
	float position_diff = (pose1.translation() - pose2.translation()).norm();
	Eigen::AngleAxisf rotation_diff_mat((pose1.rotation().inverse().lazyProduct(pose2.rotation()).eval()));

	float rotation_diff_angle = fabsf(rotation_diff_mat.angle());

	if (position_diff < clustering_position_diff_threshold_ && rotation_diff_angle < clustering_rotation_diff_threshold_)
		return true;
	else return false;
}


//////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::B2BTL_MEAMRegistration::poseWithVotesCompareFunction(const typename pcl::B2BTL_MEAMRegistration::PoseWithVotes &a,
	const typename pcl::B2BTL_MEAMRegistration::PoseWithVotes &b)
{
	return (a.votes > b.votes);
}


//////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::B2BTL_MEAMRegistration::clusterVotesCompareFunction(const std::pair<size_t, unsigned int> &a,
	const std::pair<size_t, unsigned int> &b)
{
	return (a.second > b.second);
}

void
pcl::B2BTL_MEAMRegistration::computeFinalPoses(typename pcl::B2BTL_MEAMRegistration::PoseWithVotesList &result)
{
	PoseWithVotesList voted_poses;
	std::vector <std::vector <unsigned int> > accumulator_array;
	accumulator_array.resize(search_method_->EAM_end_indices.back());

	size_t aux_size = static_cast<size_t> (floor(2 * M_PI / search_method_->getAngleDiscretizationStep()));
	for (size_t i = 0; i < search_method_->EAM_end_indices.back(); ++i)
	{
		std::vector<unsigned int> aux(aux_size);
		accumulator_array[i] = aux;
	}

	PCL_INFO("Accumulator array size: %u x %u.\n", accumulator_array.size(), accumulator_array.back().size());

	// Consider every <scene_reference_point_sampling_rate>-th point as the reference point => fix s_r
	float f1, f2, f3, f4;
	for (size_t scene_reference_index = 0; scene_reference_index < target_->points.size(); scene_reference_index += scene_reference_point_sampling_rate_)
	{
		Eigen::Vector3f scene_reference_point = target_->points[scene_reference_index].getVector3fMap(),
			scene_reference_normal = target_->points[scene_reference_index].getNormalVector3fMap();

		float rotation_angle_sg = acosf(scene_reference_normal.dot(Eigen::Vector3f::UnitX()));
		bool parallel_to_x_sg = (scene_reference_normal.y() == 0.0f && scene_reference_normal.z() == 0.0f);
		Eigen::Vector3f rotation_axis_sg = (parallel_to_x_sg) ? (Eigen::Vector3f::UnitY()) : (scene_reference_normal.cross(Eigen::Vector3f::UnitX()).normalized());
		Eigen::AngleAxisf rotation_sg(rotation_angle_sg, rotation_axis_sg);
		Eigen::Affine3f transform_sg(Eigen::Translation3f(rotation_sg * ((-1) * scene_reference_point)) * rotation_sg);

		// For every other point in the scene => now have pair (s_r, s_i) fixed
		std::vector<int> indices;

		std::vector<float> distances;
		scene_search_tree_->setInputCloud(target_);
		scene_search_tree_->radiusSearch(target_->points[scene_reference_index],
			search_method_->getModelDiameter() / 2,
			indices,
			distances);

		for (size_t i = 0; i < indices.size(); ++i)
			//    for(size_t i = 0; i < target_->points.size (); ++i)
		{
			//size_t scene_point_index = i;
			size_t scene_point_index = indices[i];
			if (scene_reference_index != scene_point_index)
			{
				if (pcl::computeB2BTL_MEAMPairFeature(target_->points[scene_reference_index].getVector4fMap(),
					target_->points[scene_reference_index].getNormalVector4fMap(),
					target_->points[scene_point_index].getVector4fMap(),
					target_->points[scene_point_index].getNormalVector4fMap(),
					f1, f2, f3, f4))
				{
					std::vector<std::pair<size_t, size_t> > nearest_indices;
					search_method_->nearestNeighborSearch(f1, f2, f3, f4, nearest_indices);

					// Compute alpha_s angle
					Eigen::Vector3f scene_point = target_->points[scene_point_index].getVector3fMap();

					Eigen::Vector3f scene_point_transformed = transform_sg * scene_point;
					float alpha_s = atan2f(-scene_point_transformed(2), scene_point_transformed(1));
					if (sin(alpha_s) * scene_point_transformed(2) < 0.0f)
						alpha_s *= (-1);
					alpha_s *= (-1);
					// Go through point pairs in the model with the same discretized feature
					for (std::vector<std::pair<size_t, size_t> >::iterator v_it = nearest_indices.begin(); v_it != nearest_indices.end(); ++v_it)
					{
						int viewpoint = 0;
						for (; viewpoint < search_method_->EAM_end_indices.size(); ++viewpoint)
						{
							if (v_it->first < search_method_->EAM_end_indices[viewpoint + 1])
								break;
						}

						size_t model_reference_index = v_it->first - search_method_->EAM_end_indices[viewpoint],
							model_point_index = v_it->second - search_method_->EAM_end_indices[viewpoint];
						// Calculate angle alpha = alpha_m - alpha_s
						float alpha = search_method_->alpha_m_EAM[viewpoint][model_reference_index][model_point_index] - alpha_s;

						if (alpha < -M_PI)
							alpha += (M_PI * 2.0);
						if (alpha > M_PI)
							alpha -= (M_PI * 2.0);

						unsigned int alpha_discretized = static_cast<unsigned int> (floor(alpha + M_PI) / search_method_->getAngleDiscretizationStep());
						accumulator_array[v_it->first][alpha_discretized] ++;
					}
				}
				else PCL_ERROR("[pcl::B2BTL_MEAMRegistration::computeTransformation] Computing pair feature vector between points %u and %u went wrong.\n", scene_reference_index, scene_point_index);
			}
		}

		size_t max_votes_i = 0, max_votes_j = 0;
		unsigned int max_votes = 0;

		for (size_t i = 0; i < accumulator_array.size(); ++i)
			for (size_t j = 0; j < accumulator_array.back().size(); ++j)
			{
				if (accumulator_array[i][j] > max_votes)
				{
					max_votes = accumulator_array[i][j];
					max_votes_i = i;
					max_votes_j = j;
				}
				// Reset accumulator_array for the next set of iterations with a new scene reference point
				accumulator_array[i][j] = 0;
			}

		int viewpoint = 0;
		for (; viewpoint < search_method_->EAM_end_indices.size(); ++viewpoint)
		{
			if (max_votes_i < search_method_->EAM_end_indices[viewpoint + 1])
				break;
		}


		Eigen::Vector3f model_reference_point = input_list[viewpoint]->points[max_votes_i - search_method_->EAM_end_indices[viewpoint]].getVector3fMap(),
			model_reference_normal = input_list[viewpoint]->points[max_votes_i - search_method_->EAM_end_indices[viewpoint]].getNormalVector3fMap();

		float rotation_angle_mg = acosf(model_reference_normal.dot(Eigen::Vector3f::UnitX()));
		bool parallel_to_x_mg = (model_reference_normal.y() == 0.0f && model_reference_normal.z() == 0.0f);
		Eigen::Vector3f rotation_axis_mg = (parallel_to_x_mg) ? (Eigen::Vector3f::UnitY()) : (model_reference_normal.cross(Eigen::Vector3f::UnitX()).normalized());
		Eigen::AngleAxisf rotation_mg(rotation_angle_mg, rotation_axis_mg);
		Eigen::Affine3f transform_mg(Eigen::Translation3f(rotation_mg * ((-1) * model_reference_point)) * rotation_mg);
		Eigen::Affine3f max_transform =
			transform_sg.inverse() *
			Eigen::AngleAxisf((static_cast<float> (max_votes_j) - floorf(static_cast<float> (M_PI) / search_method_->getAngleDiscretizationStep())) * search_method_->getAngleDiscretizationStep(), Eigen::Vector3f::UnitX()) *
			transform_mg;

		voted_poses.push_back(PoseWithVotes(max_transform, max_votes, viewpoint));
	}

	// Cluster poses for filtering out outliers and obtaining more precise results
	PoseWithVotesList results;
	clusterPoses(voted_poses, results);

	//ICP
	result.clear();
	for (size_t result_i = 0; result_i < results.size(); ++result_i)
	{
		result.push_back(results[result_i]);
	}

	transformation_ = final_transformation_ = results.front().pose.matrix();
	converged_ = true;
}
