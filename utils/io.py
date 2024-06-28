import pandas as pd
import numpy as np
import utm
import cv2

from os.path import join

class Sequence:
    
    def __init__(self, split: str, id: int, base_path="data", image_folder="images") -> None:
        
        self.id = id
        self.split = split
        self.base_path = base_path
        self.image_folder = image_folder
        self.path = join(base_path, split, f"sequence_{id}")
        
        self.df_images = pd.read_csv(join(self.path, "images.csv"))
        self.df_barometer = pd.read_csv(join(self.path, "barometric_height.csv"))

        self.df_camera = pd.merge(self.df_images, self.df_barometer, left_on="image_index", right_on="index")
        self.df_camera.drop("index", axis=1, inplace=True)
        
        print(self.df_camera["image_index"].at)

        self.df_imu = pd.read_csv(join(self.path, "imu.csv"))
        self.df_init = pd.read_csv(join(self.path, "init.csv"))
        
        if self.split == "training":
            self.df_groundtruth = pd.read_csv(join(self.path, "groundtruth.csv"))
            
        self.timestamps = self.df_imu["timestamp"]
        self.current_index = 0
        
    def get_initial_state(self):
        
        latitude, longitude = self.df_init.at[0, "latitude"], self.df_init.at[0, "longitude"]
        x, y = utm.from_latlon(latitude, longitude)[:2]
        
        altitude = self.df_init.at[0, "altitude"]
        position = np.array([x, y, altitude])
        
        velocity = np.array([
            self.df_init.at[0, "velocity.x"],
            self.df_init.at[0, "velocity.y"],
            self.df_init.at[0, "velocity.z"]
        ])
        
        orientation = np.array([
            self.df_init.at[0, "angle.x"],
            self.df_init.at[0, "angle.y"],
            self.df_init.at[0, "angle.z"]
        ])
        
        return position, velocity, orientation

    def __iter__(self):
        return self

    def __next__(self):
        
        if self.current_index >= self.timestamps.shape[0]:
            raise StopIteration
        
        timestamp = self.timestamps[self.current_index]
        
        acceleration = np.array([
            self.df_imu.at[self.current_index, "acceleration.x"],
            self.df_imu.at[self.current_index, "acceleration.y"],
            self.df_imu.at[self.current_index, "acceleration.z"]
        ])
        
        angular_velocity = np.array([
            self.df_imu.at[self.current_index, "gyroscope.x"],
            self.df_imu.at[self.current_index, "gyroscope.y"],
            self.df_imu.at[self.current_index, "gyroscope.z"]
        ])
        
        true_position = np.array([
            self.df_groundtruth.at[self.current_index, "latitude"],
            self.df_groundtruth.at[self.current_index, "longitude"],
            self.df_groundtruth.at[self.current_index, "altitude"]
        ])
        
        camera_mask = (self.df_camera["timestamp"] == timestamp)
        image, barometric_height = None, None
        
        if camera_mask.any():
            camera_info = self.df_camera[camera_mask].iloc[0]
            image_path = join(self.path, self.image_folder,
                f"{int(camera_info['image_index'])}.png")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            barometric_height = camera_info["barometric_height"]

        self.current_index += 1
        
        return timestamp, acceleration, angular_velocity, image, barometric_height, true_position

if __name__ == "__main__":
    
    first_sequence = Sequence("training", 1)
    initial_state = first_sequence.get_initial_state()
    
    print("initial state:", initial_state)
    
    for ts, acc, ang_vel, img, baro_height, gt in first_sequence:
        
        print("timestamp:",ts)
        print("acceleration:", acc)
        print("angular velocity:", ang_vel)
        print("barometric height:", baro_height)
        print("groundtruth:", gt)
        
        if img is not None:
            cv2.imshow("image", img)
            cv2.waitKey(0)
