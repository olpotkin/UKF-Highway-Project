#include "ukf.h"
#include "Eigen/Dense"


/// Initializes Unscented Kalman filter
UKF::UKF() {
  useLidar = true;            // If this is false, Lidar measurements will be ignored (except for init)
  useRadar = true;            // If this is false, Radar measurements will be ignored (except for init)

  x_ = Eigen::VectorXd(5);    // Initial state vector
  P_ = Eigen::MatrixXd(5, 5); // Initial covariance matrix

  std_a_ = 1;                 // Process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 1;             // Process noise standard deviation yaw acceleration in rad/s^2
  
  /// DO NOT MODIFY measurement noise values below.
  /// These are provided by the sensor manufacturer.
  std_laspx_  = 0.15;  // Laser measurement noise standard deviation position1 in m
  std_laspy_  = 0.15;  // Laser measurement noise standard deviation position2 in m
  std_radr_   = 0.3;   // Radar measurement noise standard deviation radius in m
  std_radphi_ = 0.03;  // Radar measurement noise standard deviation angle in rad
  std_radrd_  = 0.3;   // Radar measurement noise standard deviation radius change in m/s
  /// End DO NOT MODIFY section for measurement noise values

  /// TODO: Complete the initialization. See ukf.h for other member properties.
  /// Hint: one or more values initialized above might be wildly off...
  n_x_       = 5;                                     // Init state dimension
  n_aug_     = n_x_ + 2;                              // Init augmented state dimension
  lambda_    = 3 - n_aug_;                            // Init sigma point spreading parameter
  Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1); // Init predicted sigma points matrix
  weights_   = Eigen::VectorXd(2 * n_aug_ + 1);       // Init weights of sigma points

  double weight_0 = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);

  weights_(0) = weight_0;                      // Init weights of sigma points
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = weight;
  }
}


/// Destructor
UKF::~UKF() {}


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /// TODO: Complete this function!
  /// Make sure you switch between lidar and radar measurements.
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];

      x_ << px, py, 0, 0, 0;
      P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
            0, std_laspy_ * std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Set the state with the initial location and zero velocity
      double rho    = meas_package.raw_measurements_[0];
      double phi    = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      x_ << rho * cos(phi), rho * sin(phi), rho_dot, phi, 0;
      P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
            0, std_radphi_ * std_radphi_, 0, 0, 0,
            0, 0, std_radrd_ * std_radrd_, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !useLidar) return;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !useRadar) return;

  // Compute the time elapsed between the current and previous measurements
  // dt - in seconds
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // Do prediction
  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
}


void UKF::Prediction(double delta_t) {
  /// TODO: Complete this function! Estimate the object's location.
  /// Modify the state vector, x_. Predict sigma points, the state, and the state covariance matrix.
  Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd(n_aug_, 2 * n_aug_ + 1);

  AugmentedSigmaPoints(Xsig_aug);

  SigmaPointPrediction(Xsig_aug, delta_t, Xsig_pred_);

  PredictMeanAndCovariance(Xsig_pred_, x_, P_);
}


void UKF::PredictRadarMeasurement(
  Eigen::MatrixXd& Zsig,
  Eigen::VectorXd& z_pred,
  Eigen::MatrixXd& S)
{
  // Set measurement dimension.
  // Radar can measure:
  // 1. r
  // 2. phi
  // 3. r_dot
  int measDim = 3;

  // Transform sigma points into measurement space
  // 2n+1 simga points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
    Zsig(1, i) = atan2(p_y, p_x);                                     // phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
  }

  // Mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Covariance matrix S
  S.fill(0.0);
  // 2n+1 simga points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    normalizeAngle(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  Eigen::MatrixXd R = Eigen::MatrixXd(measDim, measDim);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;

  S = S + R;
}


void UKF::PredictLidarMeasurement(
  Eigen::MatrixXd& Zsig,
  Eigen::VectorXd& z_pred,
  Eigen::MatrixXd& S)
{
  // Set measurement dimension.
  int measDim = 2;

  // Transform sigma points into measurement space
  // 2n+1 simga points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Set measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }

  // Mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;

    normalizeAngle(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  Eigen::MatrixXd R = Eigen::MatrixXd(measDim, measDim);
  R << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

  S = S + R;
}


void UKF::UpdateState(
  MeasurementPackage& meas_package,
  Eigen::VectorXd&    z_pred,
  Eigen::MatrixXd&    S,
  Eigen::MatrixXd&    Zsig,
  Eigen::VectorXd&    x,
  Eigen::MatrixXd&    P)
{
  Eigen::VectorXd z = meas_package.raw_measurements_;
  int measDim = z.size();

  // Matrix for cross-correlation Tc
  Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, measDim);

  // Calculate cross-correlation matrix
  Tc.fill(0.0);
  // 2n+1 simga points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    normalizeAngle(z_diff(1));

    // State difference
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x;
    normalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // K - Kalman gain
  Eigen::MatrixXd K = Tc * S.inverse();

  // Residual
  Eigen::VectorXd z_diff = z - z_pred;
  normalizeAngle(z_diff(1));

  // Update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K * S * K.transpose();
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /// TODO: Complete this function! Use lidar data to update the belief
  /// about the object's position. Modify the state vector, x_, and covariance, P_.
  /// You can also calculate the lidar NIS, if desired.
  int measDim = 2;
  Eigen::VectorXd z_pred(measDim);
  Eigen::MatrixXd S(measDim, measDim);

  Eigen::MatrixXd Zsig = Eigen::MatrixXd(measDim, 2 * n_aug_ + 1);

  PredictLidarMeasurement(Zsig, z_pred, S);
  UpdateState(meas_package, z_pred, S, Zsig, x_, P_);
}


void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /// TODO: Complete this function! Use radar data to update the belief
  /// about the object's position. Modify the state vector, x_, and covariance, P_.
  /// You can also calculate the radar NIS, if desired.
  int measDim = 3;
  Eigen::VectorXd z_pred(measDim);
  Eigen::MatrixXd S(measDim, measDim);

  Eigen::MatrixXd Zsig = Eigen::MatrixXd(measDim, 2 * n_aug_ + 1);

  PredictRadarMeasurement(Zsig, z_pred, S);
  UpdateState(meas_package, z_pred, S, Zsig, x_, P_);
}


void UKF::AugmentedSigmaPoints(Eigen::MatrixXd& Xsig_aug)
{
  // Augmented mean vector
  Eigen::VectorXd x_aug = Eigen::VectorXd(n_aug_);

  // Create augmented state covariance
  Eigen::MatrixXd P_aug = Eigen::MatrixXd(n_aug_, n_aug_);

  // Init augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_)      = 0;
  x_aug(n_x_ + 1)  = 0;

  // Init augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_)               = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1)       = std_yawdd_ * std_yawdd_;

  // Square root matrix
  Eigen::MatrixXd L = P_aug.llt().matrixL();

  // Init augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}


void UKF::SigmaPointPrediction(
  Eigen::MatrixXd& Xsig_aug,
  double           delta_t,
  Eigen::MatrixXd& Xsig_pred)
{
  // Predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Extract values for better readability
    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // Predicted state values
    double px_p;
    double py_p;
    double eps = 0.001;

    // Avoid division by zero
    if (fabs(yawd) > eps) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p    = v;
    double yaw_p  = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p  = v_p + nu_a * delta_t;

    yaw_p  = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // Write predicted sigma point into right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }
}


void UKF::PredictMeanAndCovariance(
  Eigen::MatrixXd& Xsig_pred,
  Eigen::VectorXd& x,
  Eigen::MatrixXd& P)
{
  // Predicted state mean
  x.fill(0.0);
  // Iterate over sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x = x + weights_(i) * Xsig_pred.col(i);
  }

  // Predicted state covariance matrix
  P.fill(0.0);
  // Iterate over sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // State difference
    Eigen::VectorXd x_diff = Xsig_pred.col(i) - x;
    normalizeAngle(x_diff(3));

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }
}


void UKF::normalizeAngle(double& angle) {
  while (angle < -M_PI) {
    angle += 2. * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2. * M_PI;
  }
}
