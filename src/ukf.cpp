#include "ukf.h"
#include "Eigen/Dense"


/// Initializes Unscented Kalman filter
UKF::UKF() {

  useLidar = true;                    // If this is false, Lidar measurements will be ignored (except during init)
  useRadar = true;                    // If this is false, Radar measurements will be ignored (except during init)

  n_x_ = 5;                           // State vector dimension
  x_   = Eigen::VectorXd(n_x_);       // Initial state vector
  P_   = Eigen::MatrixXd(n_x_, n_x_); // Initial covariance matrix

  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0.0225, 0,
        0, 0, 0, 0, 0.0225;

  std_a_     = 0.8;            // Process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 0.5;            // Process noise standard deviation yaw acceleration in rad/s^2

  /// DO NOT MODIFY measurement noise values below.
  /// These are provided by the sensor manufacturer.
  std_laspx_  = 0.15;          // Laser measurement noise standard deviation position1 in m
  std_laspy_  = 0.15;          // Laser measurement noise standard deviation position2 in m
  std_radr_   = 0.3;           // Radar measurement noise standard deviation radius in m
  std_radphi_ = 0.03;          // Radar measurement noise standard deviation angle in rad
  std_radrd_  = 0.3;           // Radar measurement noise standard deviation radius change in m/s
  /// End DO NOT MODIFY section for measurement noise values

  /// TODO: Complete the initialization. See ukf.h for other member properties.
  /// Hint: one or more values initialized above might be wildly off...
  n_aug_   = n_x_ + 2;         // Augment state dimension
  lambda_  = 3 - n_aug_;
  time_us_ = 0.0;

  // Initialize weights
  weights_ = Eigen::VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Sigma point prediction
  Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Initialize measurement noise Covariance matrices for Radar and Lidar
  R_radar_ = Eigen::MatrixXd(3, 3);
  R_lidar_ = Eigen::MatrixXd(2, 2);

  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}


/// Destructor
UKF::~UKF() {}


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /// TODO: Complete this function!
  /// Make sure you switch between lidar and radar measurements.
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho     = meas_package.raw_measurements_[0];
      double phi     = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];
      double x       = rho * cos(phi);
      double y       = rho * sin(phi);
      double vx      = rho_dot * cos(phi);
      double vy      = rho_dot * sin(phi);
      double v       = sqrt(vx * vx + vy * vy);
      x_ << x , y, v, 0, 0;
    }
    else {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    // Init is done: no need to Predict or Update
    is_initialized_ = true;

    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_  = meas_package.timestamp_;

  // Do Prediction
  Prediction(dt);

  // Do Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && useRadar)
    UpdateRadar(meas_package);
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && useLidar)
    UpdateLidar(meas_package);
}


void UKF::Prediction(double delta_t) {
  /// TODO: Complete this function! Estimate the object's location.
  /// Modify the state vector, x_. Predict sigma points, the state, and the state covariance matrix.
  // Generate augmented sigma points
  Eigen::VectorXd x_aug    = Eigen::VectorXd(n_aug_);
  Eigen::MatrixXd P_aug    = Eigen::MatrixXd(n_aug_, n_aug_);
  Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd(n_aug_, 2*n_aug_+1);

  x_aug.head(5) = x_;
  x_aug(5)      = 0;
  x_aug(6)      = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5)               = std_a_ * std_a_;
  P_aug(6, 6)               = std_yawdd_ * std_yawdd_;

  Eigen::MatrixXd L = P_aug.llt().matrixL(); // Square root matrix
  Xsig_aug.col(0)   = x_aug;                 // Augmented sigma points

  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x     = Xsig_aug(0, i);
    double p_y     = Xsig_aug(1, i);
    double v       = Xsig_aug(2, i);
    double yaw     = Xsig_aug(3, i);
    double yawd    = Xsig_aug(4, i);
    double nu_a    = Xsig_aug(5, i);
    double nu_yawd = Xsig_aug(6, i);

    // Predicted state values
    double px_p;
    double py_p;
    double v_p;
    double yaw_p;
    double yawd_p;

    // Prevent division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }
    v_p    = v;
    yaw_p  = yaw + yawd * delta_t;
    yawd_p = yawd;

    // Add noise
    px_p   = px_p   + 0.5     * nu_a    * delta_t * delta_t * cos(yaw);
    py_p   = py_p   + 0.5     * nu_a    * delta_t * delta_t * sin(yaw);
    v_p    = v_p    + nu_a    * delta_t;
    yaw_p  = yaw_p  + 0.5     * nu_yawd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawd * delta_t;

    // Predicted sigma points
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // Predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  }

  // Predict state Covariance
  P_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Normalize angle
    while(x_diff(3) > M_PI) {
      x_diff(3) -= 2. * M_PI;
    }
    while(x_diff(3) < -M_PI) {
      x_diff(3) += 2. * M_PI;
    }

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /// TODO: Complete this function! Use lidar data to update the belief
  /// about the object's position. Modify the state vector, x_, and covariance, P_.
  /// You can also calculate the lidar NIS, if desired.

  // Extract measurement
  Eigen::VectorXd z_ = meas_package.raw_measurements_;

  int n_z_      = 2;
  Eigen::MatrixXd Zsig =Eigen::MatrixXd(n_z_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  Eigen::VectorXd z_pred_= Eigen::VectorXd(n_z_);
  z_pred_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred_ = z_pred_ + weights_(i) * Zsig.col(i);
  }

  // Covariance of predicted measurement
  Eigen::MatrixXd S = Eigen::MatrixXd(n_z_, n_z_);
  S.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred_;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S = S + R_lidar_;                   // Add noise

  // Update UKF
  // Cross-correlation matrix
  Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred_;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate K -  Kalman gain
  Eigen::MatrixXd K = Tc * S.inverse();

  // Update State Mean and Covariance
  Eigen::VectorXd z_diff = z_ - z_pred_;
  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /// TODO: Complete this function! Use radar data to update the belief
  /// about the object's position. Modify the state vector, x_, and covariance, P_.
  /// You can also calculate the radar NIS, if desired.
  Eigen::VectorXd z_ = meas_package.raw_measurements_;

  int n_z_      = 3;
  Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x  = Xsig_pred_(0, i);
    double p_y  = Xsig_pred_(1, i);
    double v    = Xsig_pred_(2, i);
    double yaw  = Xsig_pred_(3, i);
    double yawd = Xsig_pred_(4, i);

    double vx = cos(yaw)*v;
    double vy = sin(yaw)*v;

    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                     // r
    Zsig(1, i) = atan2(p_y, p_x);                             // phi
    Zsig(2, i) = (p_x*vx + p_y*vy)/(sqrt(p_x*p_x + p_y*p_y)); // r_dot
  }

  // Calculate mean predicted measurement
  Eigen::VectorXd z_pred_ = Eigen::VectorXd(n_z_);
  z_pred_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred_ = z_pred_ + weights_(i) * Zsig.col(i);
  }

  // Calculate Covariance of predicted measurement
  Eigen::MatrixXd S = Eigen::MatrixXd(n_z_, n_z_);
  S.fill(0.0);

  for (int i = 0; i < 2*n_aug_+1; i++) {
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred_;

    while (z_diff(1) > M_PI) {
      z_diff(1) -= 2. * M_PI;
    }
    while (z_diff(1) < -M_PI) {
      z_diff(1) += 2. * M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S = S + R_radar_;

  // Update UKF
  Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while(x_diff(3) > M_PI) {
      x_diff(3) -= 2. * M_PI;
    }
    while(x_diff(3) < -M_PI) {
      x_diff(3) += 2. * M_PI;
    }

    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred_;

    while(z_diff(1) > M_PI) {
      z_diff(1) -= 2. * M_PI;
    }
    while(z_diff(1) < -M_PI) {
      z_diff(1) += 2. * M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  Eigen::MatrixXd K = Tc * S.inverse();  // Calculate K - Kalman gain
  Eigen::VectorXd z_diff = z_ - z_pred_; // Update state mean and covariance

  while(z_diff(1) > M_PI) {
    z_diff(1) -= 2. * M_PI;
  }
  while(z_diff(1) < -M_PI) {
    z_diff(1) += 2. * M_PI;
  }

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
