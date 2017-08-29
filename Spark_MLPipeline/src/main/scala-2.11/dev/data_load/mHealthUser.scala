package dev.data_load

/**
  * Created by lucieburgess on 13/08/2017.
  */
case class MHealthUser(acc_Chest_X: Double, acc_Chest_Y: Double, acc_Chest_Z: Double,
                       ecg_1: Double, ecg_2: Double,
                       acc_Ankle_X: Double, acc_Ankle_Y: Double, acc_Ankle_Z: Double,
                       gyro_Ankle_X: Double, gyro_Ankle_Y: Double, gyro_Ankle_Z: Double,
                       magno_Ankle_X: Double, magno_Ankle_Y: Double, magno_Ankle_Z: Double,
                       acc_Arm_X: Double, acc_Arm_Y: Double, acc_Arm_Z: Double,
                       gyro_Arm_X: Double, gyro_Arm_Y: Double, gyro_Arm_Z: Double,
                       magno_Arm_X: Double, magno_Arm_Y: Double, magno_Arm_Z: Double,
                       activityLabel: Int)

case object MHealthUser {
  final val allowedInputCols: Array[String] = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "ecg_1", "ecg_2",
    "acc_Ankle_X", "acc_Ankle_Y", "acc_Ankle_Z", "gyro_Ankle_X", "gyro_Ankle_Y", "gyro_Ankle_Z",
    "magno_Ankle_X", "magno_Ankle_Y", "magno_Ankle_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z",
    "gyro_Arm_X", "gyro_Arm_Y", "gyro_Arm_Z", "magno_Arm_X", "magno_Arm_Y", "magno_Arm_Z", "activityLabel")
}