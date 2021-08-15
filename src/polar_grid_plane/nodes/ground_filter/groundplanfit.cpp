/*
    @file groundplanfit.cpp
    @brief ROS Node for ground plane fitting
    This is a ROS node to perform ground plan fitting.
    Implementation accoriding to <Fast Segmentation of 3D Point Clouds: A Paradigm>
    In this case, it's assumed that the x,y axis points at sea-level,
    and z-axis points up. The sort of height is based on the Z-axis value.
    @author Vincent Cheung(VincentCheungm)
    @bug Sometimes the plane is not fit.
*/

#include <iostream>
#include <list>
#include <vector>
// For disable PCL complile lib, to use PointXYZIR    
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <omp.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

//Customed Point Struct for holding clustered points
namespace scan_line_run
{
  /** Euclidean Velodyne coordinate, including intensity and ring number, and label. */
  struct PointXYZIRL
  {
    PCL_ADD_POINT4D;                    // quad-word XYZ
    float    intensity;                 ///< laser intensity reading
    uint16_t ring;                      ///< laser ring number
    uint16_t label;                     ///< point label
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
  } EIGEN_ALIGN16;

}; // namespace scan_line_run

#define SLRPointXYZIRL scan_line_run::PointXYZIRL
#define RUN pcl::PointCloud<SLRPointXYZIRL>
// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(scan_line_run::PointXYZIRL,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (uint16_t, label, label))

// using eigen lib
#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;
using namespace std;

int sequence, counter;
float total_accuracy;
std_msgs::Header _velodyne_header;
std::string point_topic_, horizontal_seg_str, vertical_seg_str, vertical_dist_str, num_iter_str, scene_str;
std::ofstream outfile;
pcl::PointCloud<pcl::PointXYZL>::Ptr g_seeds_pc(new pcl::PointCloud<pcl::PointXYZL>());
pcl::PointCloud<pcl::PointXYZL>::Ptr g_all_seeds_pc(new pcl::PointCloud<pcl::PointXYZL>());
pcl::PointCloud<pcl::PointXYZL>::Ptr segmented_ground_pc(new pcl::PointCloud<pcl::PointXYZL>());
pcl::PointCloud<pcl::PointXYZL>::Ptr segmented_not_ground_pc(new pcl::PointCloud<pcl::PointXYZL>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr g_color_pc(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<SLRPointXYZIRL>::Ptr g_all_pc(new pcl::PointCloud<SLRPointXYZIRL>());

visualization_msgs::MarkerArray marker_array;


/*
    @brief Compare function to sort points. Here use z axis.
    @return z-axis accent
*/
bool point_cmp(pcl::PointXYZL a, pcl::PointXYZL b){
    return a.z<b.z;
}

/*
    @brief Ground Plane fitting ROS Node.
    @param Velodyne Pointcloud topic.
    @param Sensor Model.
    @param Sensor height for filtering error mirror points.
    @param Num of segment, iteration, LPR
    @param Threshold of seeds distance, and ground plane distance
    
    @subscirbe:/velodyne_points
    @publish:/points_no_ground, /points_ground
*/

class PointContainer 
{
     
     public:
      inline bool IsEmpty() const { return _points.empty(); }
      inline std::vector<size_t>& points() { return _points; }
      inline const std::vector<size_t>& points() const { return _points; }

     private:
      std::vector<size_t> _points;
};

class GroundPlaneFit
{
    using PointColumn = std::vector<PointContainer>;
    using PointMatrix = std::vector<PointColumn>;
public:
    GroundPlaneFit();
private:
    ros::NodeHandle node_handle_;
    ros::Subscriber points_node_sub_;
    ros::Publisher ground_points_pub_;
    ros::Publisher groundless_points_pub_;
    ros::Publisher color_points_pub_;
    ros::Publisher lowest_point_pub_;
    ros::Publisher seeds_pub_;
    ros::Publisher ego_points_pub_;
    ros::Publisher all_points_pub_;
    ros::Publisher accuracy_pub_;
    ros::Publisher marker_pub_;


    int sensor_model_;
    double sensor_height_;
    int num_seg_;
    int num_iter_;
    int num_lpr_;
    int m;
    int n;
    int seg_m;
    int seg_n;
    int cube_counter;
    double th_seeds_;
    double th_dist_;
    bool reasonable_normal;
    visualization_msgs::MarkerArray line_array;


    void velodyne_callback_(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);
    void estimate_plane(int);
    void horizontal_seg(pcl::PointCloud<pcl::PointXYZI>, vector<vector <PointContainer>>& data);
    void vertical_seg(void);
    void polar_grid_map(void);
    void extract_ground(pcl::PointCloud<pcl::PointXYZL>);
    void extract_initial_seeds(const pcl::PointCloud<pcl::PointXYZL>& p_sorted);
    void remove_ego_points(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, 
                                 pcl::PointCloud<pcl::PointXYZI>&, 
                                 pcl::PointCloud<pcl::PointXYZI>&);

    void publishCloud(const ros::Publisher*, const pcl::PointCloud<pcl::PointXYZI>::Ptr);
    void publishCloudRGB(const ros::Publisher*, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr);

    // Model parameter for ground plane fittings
    // The ground plane model is: ax+by+cz+d=0
    // Here normal:=[a,b,c], d=d
    // th_dist_d_ = threshold_dist - d 
    float d_;
    MatrixXf normal_;
    float th_dist_d_;

    vector<int> ground_index;
};    



/*
    @brief Constructor of GPF Node.
    @return void
*/
GroundPlaneFit::GroundPlaneFit():node_handle_("~"){
    // Init ROS related
    ROS_INFO("Inititalizing Ground Plane Fitter...");
    // node_handle_.param<std::string>("point_topic", point_topic_, "/lidarseg");
    // ROS_INFO("Input Point Cloud: %s", point_topic_.c_str());

    node_handle_.param("sensor_model", sensor_model_, 32);
    ROS_INFO("Sensor Model: %d", sensor_model_);

    node_handle_.param("sensor_height", sensor_height_, 2.5);
    ROS_INFO("Sensor Height: %f", sensor_height_);

    node_handle_.param("num_seg", num_seg_, 3);
    ROS_INFO("Num of Segments: %d", num_seg_);

    // node_handle_.param("num_iter", num_iter_, 3);
    num_iter_ = std::stoi(num_iter_str);
    ROS_INFO("Num of Iteration: %d", num_iter_);

    node_handle_.param("num_lpr", num_lpr_, 20);
    ROS_INFO("Num of LPR: %d", num_lpr_);

    node_handle_.param("th_seeds", th_seeds_, 0.4);
    ROS_INFO("Seeds Threshold: %f", th_seeds_);

    node_handle_.param("th_dist", th_dist_, 0.2);
    ROS_INFO("Distance Threshold: %f", th_dist_);

    // Listen to velodyne topic
    points_node_sub_ = node_handle_.subscribe(point_topic_, 300, &GroundPlaneFit::velodyne_callback_, this);
    
    // Publish Init
    std::string no_ground_topic, ground_topic;
    node_handle_.param<std::string>("no_ground_point_topic", no_ground_topic, "/nonground_points");
    ROS_INFO("No Ground Output Point Cloud: %s", no_ground_topic.c_str());
    node_handle_.param<std::string>("ground_point_topic", ground_topic, "/ground_points");
    ROS_INFO("Only Ground Output Point Cloud: %s", ground_topic.c_str());
    groundless_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>(no_ground_topic, 100, true);
    ground_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>(ground_topic, 300);
    color_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/plane_color_points", 100, true);
    // all_points_pub_ =  node_handle_.advertise<sensor_msgs::PointCloud2>("/all_points", 100);
    // lowest_point_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/lowest_points", 100, true);
    // seeds_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/seeds_points", 100, true);
    // ego_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/ego_points", 100, true);
    // accuracy_pub_ =  node_handle_.advertise<geometry_msgs::PointStamped>("/plane_accuracy", 100);
    // marker_pub_ = node_handle_.advertise<visualization_msgs::MarkerArray>("/marker_array", 100, true);

    m = std::stoi(horizontal_seg_str);
    n = std::stoi(vertical_seg_str);
}

void GroundPlaneFit::polar_grid_map()
{
  
  double angle = 2*M_PI/m;
  double grid_dist = stod(vertical_dist_str);
  cout<<"polar_grid_map"<<endl;
  int angular_resolution = 5;

  for (int r = 0; r < n; ++r) 
  {
    visualization_msgs::Marker circular_line_marker;

    circular_line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    circular_line_marker.header = _velodyne_header;
    circular_line_marker.ns = "circular_lines";
    circular_line_marker.id = r;
    circular_line_marker.action = visualization_msgs::Marker::ADD;
    circular_line_marker.scale.x = 0.3;
    circular_line_marker.color.r = 242.0/255.0;
    circular_line_marker.color.g = 92.0/255.0;
    circular_line_marker.color.b = 192.0/255.0;
    circular_line_marker.color.a = 0.7;

    for (int c = 0; c < m; ++c)
    {
      for(int k = 0; k <= angular_resolution; k++)
      {
        geometry_msgs::Point c_p;
        c_p.x = (r+1)*grid_dist*cos(angle*(c) + angle*(k)/angular_resolution);
        c_p.y = (r+1)*grid_dist*sin(angle*(c) + angle*(k)/angular_resolution);
        c_p.z = -1.9;

      }
    }

    geometry_msgs::Point f_p;
    f_p.x = (r+1)*grid_dist*cos(angle*(0));
    f_p.y = (r+1)*grid_dist*sin(angle*(0));
    f_p.z = -1.9;

    circular_line_marker.points.push_back(f_p);

    // if(r ==n-1)   
      line_array.markers.push_back(circular_line_marker);
  }

  int text_counter = 0;

  for (int c = 0; c < m; ++c) 
  {
    visualization_msgs::Marker straight_line_marker;


    straight_line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    straight_line_marker.header = _velodyne_header;
    straight_line_marker.ns = "straight_lines";
    straight_line_marker.id = c;
    straight_line_marker.action = visualization_msgs::Marker::ADD;
    straight_line_marker.scale.x = 0.3;
    straight_line_marker.color.r = 242.0/255.0;
    straight_line_marker.color.g = 92.0/255.0;
    straight_line_marker.color.b = 192.0/255.0;
    straight_line_marker.color.a = 0.7;


    for (int r = 0; r < (n+1); ++r)
    {
      geometry_msgs::Point s_p;

      if(r == 0 )
      {
        s_p.x = 0;
        s_p.y = 0;
        s_p.z = -1.9;
      }
      else
      {
        s_p.x = (r)*grid_dist*cos(angle*(c));
        s_p.y = (r)*grid_dist*sin(angle*(c));
        s_p.z = -1.9;
      }

      straight_line_marker.points.push_back(s_p);

      visualization_msgs::Marker text_marker;
      std::string text;

      text = "(" + to_string(c) + "," + to_string(r) + ")";


      text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text_marker.header = _velodyne_header;
      text_marker.ns = "text";
      text_marker.id = text_counter;
      text_marker.action = visualization_msgs::Marker::ADD;
      text_marker.text = text;
      text_marker.scale.z = 1.5;
      text_marker.color.r = 0;
      text_marker.color.g = 0;
      text_marker.color.b = 0;
      text_marker.color.a = 1.0;

      text_marker.pose.position.x = (r + 0.5)*grid_dist*cos(angle*(c));
      text_marker.pose.position.y = (r + 0.5)*grid_dist*sin(angle*(c));
      text_marker.pose.position.z = -1.0;
     
      text_marker.pose.orientation.x = 0.0;
      text_marker.pose.orientation.y = 0.0;
      text_marker.pose.orientation.z = 0.0;
      text_marker.pose.orientation.w = 1.0;


      line_array.markers.push_back(text_marker);
      text_counter++;
    }

  }

  marker_pub_.publish(line_array);
}

/*
    @brief The function to estimate plane model. The
    model parameter `normal_` and `d_`, and `th_dist_d_`
    is set here.
    The main step is performed SVD(UAV) on covariance matrix.
    Taking the sigular vector in U matrix according to the smallest
    sigular value in A, as the `normal_`. `d_` is then calculated 
    according to mean ground points.
    @param g_ground_pc:global ground pointcloud ptr.
    
*/
void GroundPlaneFit::estimate_plane(int num_iter)
{
    // Create covarian matrix in single pass.
    // TODO: compare the efficiency.
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;
    if(segmented_ground_pc->points.size() >= 3)
    {
        pcl::computeMeanAndCovarianceMatrix(*segmented_ground_pc, cov, pc_mean);
        // Singular Value Decomposition: SVD
        JacobiSVD<MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
        // use the least singular vector as normal
        normal_ = (svd.matrixU().col(2));
        // mean ground seeds value
        Eigen::Vector3f seeds_mean = pc_mean.head<3>();


        Eigen::Vector3f v1(normal_(0,0), normal_(1,0), normal_(2,0));
        Eigen::Vector3f v2(0.0, 0.0, 1.0);

        double angle_diff = acos(v1.dot(v2));
        
        if(angle_diff < 0.35)  // angle_diff < 60 degree (1.0471975512) 40(0.69813170079)
          reasonable_normal = true;
        else
          reasonable_normal = false;

        Eigen::Quaternionf out; 
        out.setFromTwoVectors(v1,v2);

        // according to normal.T*[x,y,z] = -d
        d_ = (normal_.transpose()*seeds_mean)(0,0);
        // set distance threhold to `th_dist - d`
        th_dist_d_ = th_dist_ - d_;

        if(seg_m == 39 && seg_n == 6 &&  num_iter == 2)
        {
            visualization_msgs::Marker cube_marker;

            cube_marker.type = visualization_msgs::Marker::CUBE;
            cube_marker.header = _velodyne_header;
            cube_marker.ns = "cubes";
            cube_marker.id = cube_counter;
            cube_marker.action = visualization_msgs::Marker::ADD;
            cube_marker.scale.x = 2.0;
            cube_marker.scale.y = 2.0;
            cube_marker.scale.z = th_dist_*2;
            cube_marker.color.r = 255.0/255.0;
            cube_marker.color.g = 0.0;
            cube_marker.color.b = 0.0;
            cube_marker.color.a = 0.7;

            cube_marker.pose.position.x = seeds_mean(0);
            cube_marker.pose.position.y = seeds_mean(1);
            cube_marker.pose.position.z = seeds_mean(2);
             
            cube_marker.pose.orientation.x = out.x();
            cube_marker.pose.orientation.y = out.y();
            cube_marker.pose.orientation.z = out.z();
            cube_marker.pose.orientation.w = out.w();

            line_array.markers.push_back(cube_marker);
        }
        

        cube_counter++;
        // line_array.markers.push_back(cube_marker);

    }
    else
    {
        // cout<<"No enough points for plane extraction"<<endl;
        return;
    }
}


/*
    @brief Extract initial seeds of the given pointcloud sorted segment.
    This function filter ground seeds points accoring to heigt.
    This function will set the `g_ground_pc` to `g_seed_pc`.
    @param p_sorted: sorted pointcloud
    
    @param ::num_lpr_: num of LPR points
    @param ::th_seeds_: threshold distance of seeds
    @param ::
*/
void GroundPlaneFit::extract_initial_seeds(const pcl::PointCloud<pcl::PointXYZL>& p_sorted)
{
    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;
    // Calculate the mean height value.
    for(size_t i=0;i<p_sorted.points.size() && cnt<num_lpr_;i++)
    {
        if(p_sorted.points[i].z > -3.5)
        {
            sum += p_sorted.points[i].z;
            cnt++;

            // cout<<"index: "<<p_sorted.points[i].label<<" z: "<<p_sorted.points[i].z<<endl;
        }
    }
    double lpr_height = cnt!=0?sum/cnt:0;// in case divide by 0
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    // cout<<"lpr_height + th_seeds_: "<<lpr_height + th_seeds_<<endl;
    // cout<<"lpr_height - th_seeds_: "<<lpr_height - th_seeds_<<endl;
    for(size_t i=0;i<p_sorted.points.size();i++)
    {
        if((p_sorted.points[i].z < lpr_height + th_seeds_) &&
           (p_sorted.points[i].z > lpr_height - th_seeds_))
        {
            g_seeds_pc->points.push_back(p_sorted.points[i]);
        }
        
    }
}

/*
    @brief Velodyne pointcloud callback function. The main GPF pipeline is here.
    PointCloud SensorMsg -> Pointcloud -> z-value sorted Pointcloud
    ->error points removal -> extract ground seeds -> ground plane fit mainloop
*/
void GroundPlaneFit::horizontal_seg(pcl::PointCloud<pcl::PointXYZI> laserCloudIn, vector<vector <PointContainer>>& data)
{
    double angle_div = 2*M_PI/m;
    double vert_dist = std::stod(vertical_dist_str);

    for(size_t i=0;i<laserCloudIn.points.size();i++)
    {
        const auto& point = laserCloudIn.points[i];
        double angle = atan2(point.y, point.x) + M_PI;
        double dist = sqrt(point.x * point.x + point.y * point.y);
        int hor_seg = (floor)(angle/angle_div);
        int vert_seg = (floor)(dist/vert_dist);

        if(hor_seg > m-1 || hor_seg < 0)
        {
            // cout<<"horiznotal angle larger than expected: "<<angle<<endl;
            // cout<<"limited: 0 ~ 360"<<endl;
            // cout<<"m_size: "<<data.size()<<" n_size: "<<data[0].size()<<endl;
            // cout<<"m: "<<m<<" n: "<<n<<" hor_seg: "<<hor_seg<<" vert_seg: "<<vert_seg<<endl;
            data[0][vert_seg].points().push_back(i);
            continue;
        } 

        if(vert_seg >= n-1)
        {
            cout<<"radial direction larger than expected: "<<dist<<endl;
            cout<<"limited: "<<vert_dist*(n)<<endl;
            continue;
        } 
            
        size_t index = i;
        data[hor_seg][vert_seg].points().push_back(index);
        // cout<<"push_back: "<<i<<" data["<<hor_seg<<"]"<<"["<<vert_seg<<"]: "<<data[hor_seg][vert_seg].points().size()<<endl;
    }
}

void GroundPlaneFit::extract_ground(pcl::PointCloud<pcl::PointXYZL> segmented_cloud)
{
    segmented_ground_pc->clear();
    segmented_not_ground_pc->clear();
    extract_initial_seeds(segmented_cloud);
    segmented_ground_pc = g_seeds_pc;

    // cout<<"points_index_size: "<<segmented_cloud.points.size()<<endl;

    // for(size_t m = 0; m < segmented_cloud.points.size(); m++)
    // {
    //     cout<<"points_index: "<<segmented_cloud.points[m].label<<endl;
    // }


    MatrixXf points;
    points.setZero(segmented_cloud.points.size(),3);
    int j =0;

    
    for(auto p:segmented_cloud.points)
    {
        points.row(j++)<<p.x,p.y,p.z;
    }

    for(int i = 0; i < num_iter_; i++)
    {
        // cout<<"iter: "<<i<<" points_size: "<<segmented_ground_pc->points.size()<<endl;
        estimate_plane(i);
        segmented_ground_pc->clear();
        segmented_not_ground_pc->clear();

        VectorXf result = points*normal_;

        for(int r = 0; r < result.rows(); r++)
        {
            int index = segmented_cloud.points[r].label;


            if(result[r] > d_ - th_dist_ && result[r] < d_ + th_dist_)
            {
              if(i == num_iter_-1)
              {
                if(reasonable_normal)
                  g_all_pc->points[index].label = 0u;
                else
                  g_all_pc->points[index].label = 1u;
              }

              segmented_ground_pc->points.push_back(segmented_cloud[r]);
              // cout<<"index: "<<index<<" ground"<<endl;
            }
            else
            {
              if(i == num_iter_-1)
                g_all_pc->points[index].label = 1u;

              segmented_not_ground_pc->points.push_back(segmented_cloud[r]);

            }
        }

    }
}

void GroundPlaneFit::remove_ego_points(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                                             pcl::PointCloud<pcl::PointXYZI>& out_filtered_cloud,
                                             pcl::PointCloud<pcl::PointXYZI>& ego_cloud)
{
  pcl::ExtractIndices<pcl::PointXYZI> extractor;
  extractor.setInputCloud (in_cloud_ptr);
  pcl::PointIndices indices;

  for (size_t i=0; i< in_cloud_ptr->points.size(); i++)
  {
    pcl::PointXYZI temp_point;
    temp_point = in_cloud_ptr->points[i];

    if((temp_point.x*temp_point.x + temp_point.y*temp_point.y + temp_point.z*temp_point.z) < 9 )
        indices.indices.push_back(i);
  }
  extractor.setIndices(boost::make_shared<pcl::PointIndices>(indices));
  extractor.setNegative(true);//true removes the indices, false leaves only the indices
  extractor.filter(out_filtered_cloud);
  extractor.setNegative(false);//true removes the indices, false leaves only the indices
  extractor.filter(ego_cloud);
}

void GroundPlaneFit::publishCloud(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}

void GroundPlaneFit::publishCloudRGB(const ros::Publisher* in_publisher, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header=_velodyne_header;
  in_publisher->publish(cloud_msg);
}

void GroundPlaneFit::velodyne_callback_(const sensor_msgs::PointCloud2ConstPtr& in_cloud_msg)
{
    // 1.Msg to pointcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn_org(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*in_cloud_msg, *laserCloudIn_org);
    
    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
    pcl::PointCloud<pcl::PointXYZI> ego_cloud;

    remove_ego_points(laserCloudIn_org, laserCloudIn, ego_cloud);

    cube_counter = 0;
    _velodyne_header = in_cloud_msg->header;
    marker_array.markers.clear();
    // For mark ground points and hold all points
    SLRPointXYZIRL point;
    for(size_t i=0;i<laserCloudIn.points.size();i++)
    {
        laserCloudIn.points[i].z = laserCloudIn.points[i].z;
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        point.intensity = laserCloudIn.points[i].intensity;
        // point.ring = laserCloudIn.points[i].ring;
        point.label = 0u;// 0 means uncluster
        g_all_pc->points.push_back(point);
    }

    vector<PointColumn> _data(m, vector<PointContainer>(n));
    horizontal_seg(laserCloudIn, _data);
    // cout<<"after horizontal_seg"<<endl;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            pcl::PointCloud<pcl::PointXYZL> segmented_cloud;
            vector<size_t> points_index(_data[i][j].points().size(),0);

            segmented_cloud.clear();

            for(size_t k = 0; k < _data[i][j].points().size(); k++)
            {
                size_t index = _data[i][j].points()[k];
                // cout<<"index: "<<_data[i][j].points()[k]<<endl;
                const auto& point = laserCloudIn.points[index];

                pcl::PointXYZL current_point;

                current_point.x = point.x;
                current_point.y = point.y;
                current_point.z = point.z;
                current_point.label = index;

                segmented_cloud.points.push_back(current_point);
            }

            // cout<<"m: "<<i<<" n: "<<j<<" size: "<<_data[i][j].points().size()<<endl;
            sort(segmented_cloud.points.begin(),segmented_cloud.end(),point_cmp);
            // cout<<"m: "<<i<<" n: "<<j<<" size: "<<segmented_cloud.points.size()<<endl;

            seg_m = i;
            seg_n = j;
            if(segmented_cloud.points.size() >= 3)
                extract_ground(segmented_cloud);
            else
                continue;
        }
    }
     // Calculate Accuracy and T,FP,FN
    int true_points, false_positive_points, false_negative_points;
    true_points = false_positive_points = false_negative_points =  0;

    segmented_ground_pc->clear();
    segmented_not_ground_pc->clear();

    pcl::PointCloud<pcl::PointXYZI>::Ptr ground_seg(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr nonground_seg(new pcl::PointCloud<pcl::PointXYZI>());

    for(size_t i = 0; i < g_all_pc->points.size(); i++)
    {
        int intensity = g_all_pc->points[i].intensity;
        int label = g_all_pc->points[i].label;

        pcl::PointXYZI temp_point;

        temp_point.x = g_all_pc->points[i].x;
        temp_point.y = g_all_pc->points[i].y;
        temp_point.z = g_all_pc->points[i].z;
        temp_point.intensity = g_all_pc->points[i].intensity;

        if(label == 0u && g_all_pc->points[i].z > -3.5)
            ground_seg->points.push_back(temp_point);
        else
            nonground_seg->points.push_back(temp_point);

    }


    // for(size_t i = 0; i < g_all_pc->points.size(); i++)
    // {
    //   int intensity = g_all_pc->points[i].intensity;
    //   int label = g_all_pc->points[i].label;

    //   pcl::PointXYZI temp_point;
    //   pcl::PointXYZRGB truth_point;

      // temp_point.x = g_all_pc->points[i].x;
      // temp_point.y = g_all_pc->points[i].y;
      // temp_point.z = g_all_pc->points[i].z;
      // temp_point.intensity = g_all_pc->points[i].intensity;

    //   truth_point.x = g_all_pc->points[i].x;
    //   truth_point.y = g_all_pc->points[i].y;
    //   truth_point.z = g_all_pc->points[i].z;
    //   truth_point.r = truth_point.g = truth_point.b = 0;

    //   if(scene_str == "nctu")
    //   {
    //     if(label == 0u && g_all_pc->points[i].z > -3.5)
    //     {
    //       temp_point.intensity = 0;
    //       if(intensity == 0)
    //       {
    //         true_points++;
    //         truth_point.b = 255;
    //       }
    //       else
    //         truth_point.r =255;
        
    //     }
    //     else
    //     {
    //       temp_point.intensity = 1;
    //       if(intensity == 1)
    //       {
    //         true_points++;
    //         truth_point.b = 255;
    //       }
    //       else
    //         truth_point.g =255;
    //     }
    //   }
    //   else if(scene_str == "nuscenes")
    //   {
    //     if(label == 0u)
    //     {
    //       temp_point.intensity = 0;
    //       if(intensity >= 24 && intensity <= 27)
    //       {
    //         true_points++;
    //         truth_point.b = 255;
    //       }
    //       else
    //         truth_point.g =255;
        
    //     }
    //     else
    //     {
    //       temp_point.intensity = 1;
    //       if(intensity < 24 || intensity > 27)
    //       {
    //         true_points++;
    //         truth_point.b = 255;
    //       }
    //       else
    //         truth_point.r =255;
    //     }
    //   }
      
    //   ground_seg->points.push_back(temp_point);
    //   g_color_pc->points.push_back(truth_point);

    // }

    publishCloud(&ground_points_pub_, ground_seg);
    publishCloud(&groundless_points_pub_, nonground_seg);
    // publishCloudRGB(&color_points_pub_, g_color_pc);
   
    // polar_grid_map();
    // counter++;

    // float accuracy = (float)100*(true_points)/(g_all_pc->points.size());
    // outfile<<accuracy<<endl;
    // total_accuracy += accuracy;

    // cout<<"true_points: "<<true_points<<" size: "<<g_all_pc->points.size()<<endl;
    // cout<<"Sequence: "<<counter<<" Accuracy: "<<accuracy<<" Total_accuracy: "<<total_accuracy/counter<<endl;

    ground_seg->clear();
    nonground_seg->clear();
    g_color_pc->clear();
    g_all_pc->clear();
    line_array.markers.clear();

}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "GroundPlaneFit");
    counter = 0;
    total_accuracy = 0.0;
    std::string path = ros::package::getPath("polar_grid_plane");
    std::fstream csv_config(path + "/plane_csv_name.txt");
    std::string filename;
    getline(csv_config, filename, '\n');
    getline(csv_config, point_topic_, '\n');
    getline(csv_config, horizontal_seg_str, '\n');
    getline(csv_config, vertical_seg_str, '\n');
    getline(csv_config, vertical_dist_str, '\n');
    getline(csv_config, num_iter_str, '\n');
    getline(csv_config, scene_str, '\n');
    outfile.open(path + "/csv" + filename);

    std::cout<<"sub_topic: "<<point_topic_<<std::endl;
    std::cout<<"horizontal: "<<horizontal_seg_str<<" angle division: "<<2*M_PI/stod(horizontal_seg_str)*180/M_PI<<endl;
    std::cout<<"vertical: "<<vertical_seg_str<<"  max radial distance: "<<(stoi(vertical_seg_str)) * stod(vertical_dist_str)<<endl;

    GroundPlaneFit node;
    ros::spin();

    return 0;

}