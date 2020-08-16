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

  bool is_initialized_;       // Initially set to false, set to true in first call of ProcessMeasurement
  bool useLidar;              // If this is false, Lidar measurements will be ignored (except for init)
  bool useRadar;              // If this is false, Radar measurements will be ignored (except for init)

  Eigen::VectorXd x_;         // State vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::MatrixXd P_;         // State covariance matrix
  Eigen::MatrixXd Xsig_pred_; // Predicted sigma points matrix

  long long time_us_;         // Time when the state is true, in us
  double std_a_;              // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_yawdd_;          // Process noise standard deviation yaw acceleration in rad/s^2
  double std_laspx_;          // Laser measurement noise standard deviation position1 in m
  double std_laspy_;          // Laser measurement noise standard deviation position2 in m
  double std_radr_;           // Radar measurement noise standard deviation radius in m
  double std_radphi_;         // Radar measurement noise standard deviation angle in rad
  double std_radrd_;          // Radar measurement noise standard deviation radius change in m/s

  Eigen::VectorXd weights_;   // Weights of sigma points

  int n_x_;                   // State dimension
  int n_aug_;                 // Augmented state dimension
  double lambda_;             // Sigma point spreading parameter

  /// @brief ProcessMeasurement
  /// @param meas_package The latest measurement data of either radar or laser
  void ProcessMeasurement(MeasurementPackage meas_package);

  /// @brief Prediction Predicts sigma points, the state, and the state covariance matrix
  /// @param delta_t Time between k and k+1 in s
  void Prediction(double delta_t);

  /// @brief Predict Radar measurement
  void PredictRadarMeasurement(
    Eigen::MatrixXd& Zsig,
    Eigen::VectorXd& z_pred,
    Eigen::MatrixXd& S);

  /// @brief Updates the state and the state covariance matrix using a radar measurement
  /// @param meas_package The measurement at k+1
  void UpdateRadar(MeasurementPackage meas_package);

  /// @brief Predict Lidar measurement
  void PredictLidarMeasurement(
    Eigen::MatrixXd& Zsig,
    Eigen::VectorXd& z_pred,
    Eigen::MatrixXd& S);

  /// @brief Updates the state and the state covariance matrix using a laser measurement
  /// @param meas_package The measurement at k+1
  void UpdateLidar(MeasurementPackage meas_package);

  /// @brief Update state matrix
  void UpdateState(
    MeasurementPackage& meas_package,
    Eigen::VectorXd&    z_pred,
    Eigen::MatrixXd&    S,
    Eigen::MatrixXd&    Zsig,
    Eigen::VectorXd&    x,
    Eigen::MatrixXd&    P);

  /// @brief Helper
  void AugmentedSigmaPoints(Eigen::MatrixXd& Xsig_aug);

  /// @brief Helper
  void SigmaPointPrediction(
    Eigen::MatrixXd& Xsig_aug,
    double           delta_t,
    Eigen::MatrixXd& Xsig_pred);

  /// @brief Helper
  void PredictMeanAndCovariance(
    Eigen::MatrixXd& Xsig_pred,
    Eigen::VectorXd& x_pred,
    Eigen::MatrixXd& P_pred);

  /// @brief Helper
  void normalizeAngle(double& angle);
};

#endif  // UKF_H
