#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"


class UKF {
public:
  /// @brief Constructor
  UKF();

  /// @brief Destructor
  virtual ~UKF();

  /// @brief ProcessMeasurement
  /// @param meas_package The latest measurement data of either radar or laser
  void ProcessMeasurement(MeasurementPackage meas_package);

  /// @brief Prediction Predicts sigma points, the state, and the state covariance matrix
  /// @param delta_t Time between k and k+1 in s
  void Prediction(double delta_t);

  /// @brief Updates the state and the state covariance matrix using a laser measurement
  /// @param meas_package The measurement at k+1
  void UpdateLidar(MeasurementPackage meas_package);

  /// @brief Updates the state and the state covariance matrix using a radar measurement
  /// @param meas_package The measurement at k+1
  void UpdateRadar(MeasurementPackage meas_package);

  bool is_initialized_;       // Initially set to false, set to true in first call of ProcessMeasurement
  bool useLidar;              // If this is false, laser measurements will be ignored (except for init)
  bool useRadar;              // If this is false, radar measurements will be ignored (except for init)

  Eigen::VectorXd x_;         // State vector
  Eigen::MatrixXd P_;         // State covariance matrix
  Eigen::MatrixXd Xsig_pred_; // Predicted sigma points matrix
  Eigen::VectorXd weights_;   // Weights of sigma points
  Eigen::MatrixXd R_radar_;   // Radar measurement noise covariance matrix
  Eigen::MatrixXd R_lidar_;   // Lidar measurement noise covariance matrix

  long long time_us_;         // Time when the state is true, in us
  double    std_a_;           // Process noise standard deviation longitudinal acceleration in m/s^2
  double    std_yawdd_;       // Process noise standard deviation yaw acceleration in rad/s^2
  double    std_laspx_;       // Laser measurement noise standard deviation position1 in m
  double    std_laspy_;       // Laser measurement noise standard deviation position2 in m
  double    std_radr_;        // Radar measurement noise standard deviation radius in m
  double    std_radphi_;      // Radar measurement noise standard deviation angle in rad
  double    std_radrd_;       // Radar measurement noise standard deviation radius change in m/s
  int       n_x_;             // State dimension
  int       n_aug_;           // Augmented state dimension
  double    lambda_;          // Sigma point spreading parameter
};

#endif // UKF_H
