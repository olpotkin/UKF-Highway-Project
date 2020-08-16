#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"
#include "render/render.h"
#include <pcl/io/pcd_io.h>


struct lmarker {
  double x;
  double y;

  lmarker(double setX, double setY)
    : x(setX)
    , y(setY)
  {}
};


struct rmarker {
  double rho;
  double phi;
  double rho_dot;

  rmarker(double setRho, double setPhi, double setRhoDot)
    : rho(setRho)
    , phi(setPhi)
    , rho_dot(setRhoDot)
  {}
};


class Tools {
  public:
  /// Constructor.
  Tools();

  /// Destructor.
  virtual ~Tools();

  // Members
  std::vector<Eigen::VectorXd> estimations;
  std::vector<Eigen::VectorXd> ground_truth;

  double noise(double stddev, long long seedNum);

  lmarker lidarSense(
    Car&                                    car,
    pcl::visualization::PCLVisualizer::Ptr& viewer,
    long long                               timestamp,
    bool                                    visualize);

  rmarker radarSense(
    Car&                                    car,
    Car                                     ego,
    pcl::visualization::PCLVisualizer::Ptr& viewer,
    long long                               timestamp,
    bool                                    visualize);

  void ukfResults(
    Car                                     car,
    pcl::visualization::PCLVisualizer::Ptr& viewer,
    double                                  time,
    int                                     steps);

  /// A helper method to calculate RMSE.
  Eigen::VectorXd CalculateRMSE(
    const std::vector<Eigen::VectorXd>& estimations,
    const std::vector<Eigen::VectorXd>& ground_truth);

  void savePcd(
    typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    std::string                                  file);

  pcl::PointCloud<pcl::PointXYZ>::Ptr loadPcd(std::string file);
};

#endif // TOOLS_H_
