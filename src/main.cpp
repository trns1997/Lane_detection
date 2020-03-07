/* \author Aaron Brown */
// Create simple 3d highway environment using PCL
// for exploring self-driving car sensors

/**
 * Developer: Yasen Hu
 * Date: 05/25/2019
 */

#include "render/render.h"
#include "processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "processPointClouds.cpp"

void cityBlock(pcl::visualization::PCLVisualizer::Ptr &viewer, ProcessPointClouds<pcl::PointXYZI> &point_cloud_processor, pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud)
{
    renderPointCloud(viewer, input_cloud, "InputCloud", 2);

    // Input point cloud, filter resolution, min Point, max Point
    constexpr float kFilterResolution = 0.1;
    const Eigen::Vector4f kMinPoint(0, -6.0, -3, 1);
    const Eigen::Vector4f kMaxPoint(12, 6.0, 4, 1);
    auto filter_cloud = point_cloud_processor.FilterCloud(input_cloud, kFilterResolution, kMinPoint, kMaxPoint);

    // renderPointCloud(viewer, filter_cloud, "FilteredCloud", 2);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1);

    seg.setInputCloud(filter_cloud);
    seg.segment(*inliers, *coefficients);

    pcl::PointCloud<pcl::PointXYZI>::Ptr plane(new pcl::PointCloud<pcl::PointXYZI>);
    // Fill in the cloud data
    plane->width = inliers->indices.size();
    plane->height = 1;
    plane->is_dense = false;
    plane->points.resize(plane->width * plane->height);

    for (unsigned int k = 0; k < plane->points.size(); ++k)
    {
        plane->points[k].x = filter_cloud->points[inliers->indices[k]].x;
        plane->points[k].y = filter_cloud->points[inliers->indices[k]].y;
        plane->points[k].z = filter_cloud->points[inliers->indices[k]].z;
        plane->points[k].intensity = filter_cloud->points[inliers->indices[k]].intensity;
    }

    Color clr = Color(0, 1, 0);
    renderPointCloud(viewer, plane, "plane", 5, clr);
    renderHighway(viewer, *coefficients);

    std::vector<float> diff;
    std::vector<int> ind;
    for (std::size_t i = 0; i < plane->points.size(); ++i)
    {
        diff.push_back(std::abs(plane->points[i + 1].intensity - plane->points[i].intensity));
    }

    float thresh = accumulate(diff.begin(), diff.end(), 0.0) / (plane->points.size() - 1);

    for (unsigned int i = 0; i < diff.size(); ++i)
    {
        if (diff[i] > 8 * thresh)
        {
            ind.push_back(i);
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr lane(new pcl::PointCloud<pcl::PointXYZ>);
    // Fill in the cloud data
    lane->width = ind.size();
    lane->height = 1;
    lane->is_dense = false;
    lane->points.resize(lane->width * lane->height);

    for (unsigned int j = 0; j < lane->points.size(); ++j)
    {
        lane->points[j].x = plane->points[ind[j]].x;
        lane->points[j].y = plane->points[ind[j]].y;
        lane->points[j].z = plane->points[ind[j]].z;
    }

    // Color cl = Color(0, 0, 1);
    // renderPointCloud(viewer, lane, "Lane", 10, cl);

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(lane);
    vg.setLeafSize(0.5f, 0.5f, 0.5f);
    vg.filter(*cloud_filtered);

    // Color cl1 = Color(0, 1, 0);
    // renderPointCloud(viewer, cloud_filtered, "vox", 5, cl1);

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(1.2); // 2cm
    ec.setMinClusterSize(5);
    ec.setMaxClusterSize(100);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    int j = 1;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud_filtered->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // Color clr = Color(j % 3, j % 2, j % 1);
        // renderPointCloud(viewer, cloud_cluster, std::to_string(j), 15, clr);

        // std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.1);

        seg.setInputCloud(cloud_cluster);
        seg.segment(*inliers, *coefficients);

        float xMin = 0;
        float yMin = (((coefficients->values[4])*(xMin - coefficients->values[0]))/coefficients->values[3]) + coefficients->values[1];
        float zMin = (((coefficients->values[5])*(xMin - coefficients->values[0]))/coefficients->values[3]) + coefficients->values[2];

        float xMax = 20;
        float yMax = (((coefficients->values[4])*(xMax - coefficients->values[0]))/coefficients->values[3]) + coefficients->values[1];
        float zMax = (((coefficients->values[5])*(xMax - coefficients->values[0]))/coefficients->values[3]) + coefficients->values[2];

        viewer->addLine(pcl::PointXYZ(xMin, yMin, zMin), pcl::PointXYZ(xMax, yMax, zMax), 1, 1, 1, std::to_string(j));
        j++;
    }
}

int main(int argc, char **argv)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    // CameraAngle setAngle = FPS;
    // initCamera(setAngle, viewer);

    ProcessPointClouds<pcl::PointXYZI> point_cloud_processor;
    std::vector<boost::filesystem::path> stream = point_cloud_processor.streamPcd("/home/dank-engine/3d_ws/lidar-object-detection/data/pcd/data_3");
    auto stream_iterator = stream.begin();
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud;

    // input_cloud = point_cloud_processor.loadPcd("/home/dank-engine/3d_ws/lidar-object-detection/data/pcd/data_3/03518.pcd");

    while (!viewer->wasStopped())
    {
        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();

        //     // Load pcd and run obstacle detection process
        input_cloud = point_cloud_processor.loadPcd((*stream_iterator).string());
        cityBlock(viewer, point_cloud_processor, input_cloud);

        stream_iterator++;
        // keep looping
        if (stream_iterator == stream.end())
            stream_iterator = stream.begin();

        // viewer spin
        viewer->spinOnce();
    }
}