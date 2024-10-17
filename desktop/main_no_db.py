import sys
import pandas as pd
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from pyproj import Transformer
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog
from pyvistaqt import QtInteractor
from db_connection import DatabaseConnection

class TerrainMapper(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Terrain Mapper")
        self.setGeometry(100, 100, 1000, 800)

        layout = QVBoxLayout()

        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        self.visualize_button = QPushButton("Visualize Terrain")
        self.visualize_button.clicked.connect(self.visualize_terrain)
        self.visualize_button.setEnabled(False)
        layout.addWidget(self.visualize_button)
        
        self.visualize_button_interpolation = QPushButton("Visualize Terrain (Interpolation)")
        self.visualize_button_interpolation.clicked.connect(self.visualize_terrain_interpolation)
        self.visualize_button_interpolation.setEnabled(False)
        layout.addWidget(self.visualize_button_interpolation)
        
        self.measure_button = QPushButton("Measure Distance")
        self.measure_button.clicked.connect(self.measure_distance)
        self.measure_button.setEnabled(False)
        layout.addWidget(self.measure_button)

        self.frame = QtInteractor(self)
        layout.addWidget(self.frame)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.df = None
        self.grid = None

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.df = pd.read_csv(file_path, header=0)
            self.df.columns = ['Time', 'Latitude', 'Longitude', 'Altitude']
            self.visualize_button.setEnabled(True)
            self.visualize_button_interpolation.setEnabled(True)
            self.measure_button.setEnabled(True)

    def geo_to_metric(self, lat, lon, alt):
        # Convert geographic coordinates (EPSG:4326) to metric (EPSG:3857)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        z = alt
        return x, y, z

    def interpolate_grid(self, points, values, resolution=100):
        # Generate a grid of points for interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(points[:, 0].min(), points[:, 0].max(), resolution),
            np.linspace(points[:, 1].min(), points[:, 1].max(), resolution)
        )

        # Perform interpolation on the Z values (altitude)
        grid_z = griddata(points[:, :2], values, (grid_x, grid_y), method='cubic')
        
        # Return a structured grid for pyvista
        grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
        return grid
    
    def calculate_slope(self, grid):
        gradients = grid.compute_derivatives()
        slope = np.sqrt(gradients['gradient'][:, 0]**2 + gradients['gradient'][:, 1]**2)
        return slope

    def visualize_terrain_interpolation(self):
        if self.df is not None:
            # Convert geographic coordinates to metric
            points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                               zip(self.df['Latitude'], self.df['Longitude'], self.df['Altitude'])])

            # Interpolate the terrain to create a smooth grid
            grid = self.interpolate_grid(points, self.df['Altitude'].values)

            # Create a point cloud for original points
            point_cloud = pv.PolyData(points)
            point_cloud['Altitude'] = self.df['Altitude'].values

            # Clear the previous visualization
            self.frame.clear()

            # Visualize interpolated terrain (grid) and original points
            self.frame.add_mesh(grid, scalars=grid.points[:, 2], cmap='terrain', show_edges=True, smooth_shading=True)
            self.frame.add_points(point_cloud, color='darkred', point_size=5)

            # Add grid and axes
            self.frame.show_grid()
            self.frame.show_axes()
            
            # Reset camera view
            self.frame.reset_camera()
        
    def visualize_terrain(self):
        if self.df is not None:
            points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                               zip(self.df['Latitude'], self.df['Longitude'], self.df['Altitude'])])
            points -= points.min(axis=0)

            point_cloud = pv.PolyData(points)
            point_cloud['Altitude'] = self.df['Altitude'].values

            grid = point_cloud.delaunay_2d()

            self.frame.clear()
            self.frame.add_mesh(grid, scalars='Altitude', cmap='terrain')
            self.frame.add_points(point_cloud, color='darkred', point_size=5)
            self.frame.show_grid()
            self.frame.show_axes()
            
            # Obliczanie i wyświetlanie rozmiaru obszaru
            self.frame.reset_camera()
            
    def distance_callback(self, point):
        """Callback to measure distance between points."""
        # Add logic to measure distance between picked points
        print(f"Point selected: {point}")
        if not hasattr(self, 'picked_points'):
            self.picked_points = []
        self.picked_points.append(point)
        
        if len(self.picked_points) == 2:
            # Compute the Euclidean distance between the two picked points
            p1, p2 = self.picked_points
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            print(f"Distance between points: {distance:.2f} meters")

            # Optionally, display the distance on the plot
            self.frame.add_text(f"Distance: {distance:.2f} m", position='upper_left', font_size=10)
            
            # Clear picked points for the next measurement
            self.picked_points = []


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TerrainMapper()
    window.show()
    sys.exit(app.exec())
