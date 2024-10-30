import sys
import pandas as pd
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from scipy.spatial import KDTree  # Nowy import dla metody window
from pyproj import Transformer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QComboBox, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QStatusBar, QToolBar, QMenuBar, QMenu, QStyle, QFrame, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
from pyvistaqt import QtInteractor
from db_connection import DatabaseConnection

class TerrainMapper(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terrain Mapper")
        self.setGeometry(100, 100, 1200, 800)
        
        # Inicjalizacja bazy danych
        self.db = DatabaseConnection(
            dbname="terrain_measurements",
            user="pkrypel",
            password="20122002",
            host="localhost",
            port="5433"
        )

        # Stwórz pasek menu
        self.create_menu_bar()
        
        # Stwórz pasek narzędzi
        self.create_toolbar()
        
        # Stwórz główny widget i layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Główny layout
        main_layout = QHBoxLayout(main_widget)
        
        # Panel kontrolny (lewa strona)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Panel wizualizacji (prawa strona)
        visualization_panel = self.create_visualization_panel()
        main_layout.addWidget(visualization_panel)
        
        # Ustaw proporcje między panelami
        main_layout.setStretch(0, 1)  # Panel kontrolny
        main_layout.setStretch(1, 4)  # Panel wizualizacji
        
        # Pasek statusu
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        # Inicjalizacja zmiennych
        self.df = None
        self.grid = None
        
        # Załaduj sesje
        self.load_sessions()

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # Menu File
        file_menu = menubar.addMenu("File")
        
        # Akcje w menu File
        new_action = QAction("New Session", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_session)
        
        export_action = QAction("Export Model", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_model)
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        file_menu.addAction(new_action)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # Menu View
        view_menu = menubar.addMenu("View")
        
        # Menu Tools
        tools_menu = menubar.addMenu("Tools")
        
        # Menu Help
        help_menu = menubar.addMenu("Help")

    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Przyciski na pasku narzędzi
        refresh_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Refresh", self)
        refresh_action.triggered.connect(self.load_sessions)
        toolbar.addAction(refresh_action)

    def create_control_panel(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Grupa wyboru sesji
        session_group = QGroupBox("Session Selection")
        session_layout = QVBoxLayout()
        
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self.load_session_data)
        
        self.load_button = QPushButton("Refresh Sessions")
        self.load_button.clicked.connect(self.load_sessions)
        
        session_layout.addWidget(self.session_combo)
        session_layout.addWidget(self.load_button)
        session_group.setLayout(session_layout)
        
        # Grupa parametrów wizualizacji
        visualization_group = QGroupBox("Visualization Parameters")
        visualization_layout = QVBoxLayout()
        
        # Parametry wygładzania
        smoothing_label = QLabel("Smoothing:")
        self.smoothing_method = QComboBox()
        self.smoothing_method.addItems(['laplacian', 'gaussian', 'window'])
        
        iterations_label = QLabel("Iterations:")
        self.smoothing_iterations = QSpinBox()
        self.smoothing_iterations.setRange(1, 10)
        self.smoothing_iterations.setValue(2)
        
        factor_label = QLabel("Smoothing Factor:")
        self.smoothing_factor = QDoubleSpinBox()
        self.smoothing_factor.setRange(0.1, 1.0)
        self.smoothing_factor.setSingleStep(0.1)
        self.smoothing_factor.setValue(0.5)
        
        visualization_layout.addWidget(smoothing_label)
        visualization_layout.addWidget(self.smoothing_method)
        visualization_layout.addWidget(iterations_label)
        visualization_layout.addWidget(self.smoothing_iterations)
        visualization_layout.addWidget(factor_label)
        visualization_layout.addWidget(self.smoothing_factor)
        visualization_group.setLayout(visualization_layout)
        
        # Przyciski wizualizacji
        buttons_group = QGroupBox("Visualization")
        buttons_layout = QVBoxLayout()
        
        self.visualize_button_raw = QPushButton("Visualize Raw Point Cloud")
        self.visualize_button_raw.clicked.connect(self.visualize_terrain_raw)
        self.visualize_button_raw.setEnabled(False)
        
        self.visualize_button = QPushButton("Visualize Repaired Point Cloud")
        self.visualize_button.clicked.connect(self.visualize_terrain)
        self.visualize_button.setEnabled(False)
        
        self.visualize_button_interpolation = QPushButton("Visualize Surface Model")
        self.visualize_button_interpolation.clicked.connect(self.visualize_terrain_interpolation)
        self.visualize_button_interpolation.setEnabled(False)
        
        buttons_layout.addWidget(self.visualize_button_raw)
        buttons_layout.addWidget(self.visualize_button)
        buttons_layout.addWidget(self.visualize_button_interpolation)
        buttons_group.setLayout(buttons_layout)
        
        # Dodaj wszystkie grupy do głównego layoutu
        control_layout.addWidget(session_group)
        control_layout.addWidget(visualization_group)
        control_layout.addWidget(buttons_group)
        control_layout.addStretch()
        
        # Stylizacja
        control_widget.setMaximumWidth(300)
        control_widget.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0px 5px 0px 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 5px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QLabel {
                font-weight: bold;
            }
        """)
        
        return control_widget

    def create_visualization_panel(self):
        visualization_widget = QWidget()
        visualization_layout = QVBoxLayout(visualization_widget)
        
        # Dodaj widget PyVista
        self.frame = QtInteractor(self)
        visualization_layout.addWidget(self.frame)
        
        return visualization_widget
    
    def new_session(self):
        # Implementacja tworzenia nowej sesji
        self.statusBar.showMessage("Creating new session...")
        
    def export_model(self):
        # Implementacja eksportu modelu
        if self.grid is not None:
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Export Model",
                "",
                "STL Files (*.stl);;VTK Files (*.vtk);;All Files (*.*)"
            )
            if file_name:
                try:
                    if file_name.endswith('.stl'):
                        self.grid.save(file_name)
                    elif file_name.endswith('.vtk'):
                        self.grid.save(file_name)
                    self.statusBar.showMessage(f"Model exported to {file_name}")
                except Exception as e:
                    self.statusBar.showMessage(f"Error exporting model: {str(e)}")
        else:
            self.statusBar.showMessage("No model to export")

    def load_sessions(self):
        self.session_combo.clear()
        sessions = self.db.get_sessions()
        self.session_combo.addItems(sessions)

    def load_session_data(self):
        session_name = self.session_combo.currentText()
        if session_name:
            self.df = self.db.get_session_data(session_name)
            self.visualize_button_raw.setEnabled(True)
            self.visualize_button.setEnabled(True)
            self.visualize_button_interpolation.setEnabled(True)

    def geo_to_metric(self, lat, lon, alt):
        # Convert geographic coordinates (EPSG:4326) to metric (EPSG:3857)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        z = alt
        return x, y, z
    
    def smooth_terrain(self, grid, method='laplacian', iterations=1, relaxation_factor=0.5):
        """
        Wygładza model terenu używając różnych metod.
        """
        # Konwertuj StructuredGrid na PolyData
        surf = grid.extract_surface()
        
        if method == 'laplacian':
            # Wygładzanie Laplace'a
            smoothed = surf.smooth(
                n_iter=iterations,
                relaxation_factor=relaxation_factor,
                feature_angle=75.0,
                boundary_smoothing=True
            )
            
        elif method == 'gaussian':
            # Wygładzanie gaussowskie
            smoothed = surf.smooth_gaussian(
                std_dev=1.0,
                n_iter=iterations,
                progress_bar=True
            )
            
        elif method == 'window':
            smoothed = surf.copy()
            points = smoothed.points
            
            for _ in range(iterations):
                tree = KDTree(points[:, :2])
                
                for i in range(len(points)):
                    neighbors = tree.query_ball_point(points[i, :2], r=2.0)
                    if neighbors:
                        weights = np.exp(-np.sum((points[neighbors, :2] - points[i, :2])**2, axis=1))
                        smoothed.points[i, 2] = np.average(points[neighbors, 2], weights=weights)
        
        return smoothed
    
    def visualize_terrain_interpolation(self):
        if self.df is not None:
            try:
                # Pobierz parametry wygładzania z kontrolek
                smoothing_method = self.smoothing_method.currentText()
                iterations = self.smoothing_iterations.value()
                relaxation_factor = self.smoothing_factor.value()
                
                print(f"Stosowanie parametrów wygładzania:")
                print(f"Metoda: {smoothing_method}")
                print(f"Liczba iteracji: {iterations}")
                print(f"Współczynnik wygładzania: {relaxation_factor}")

                # Konwersja współrzędnych
                points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                                zip(self.df['Latitude'], self.df['Longitude'], self.df['Altitude'])])
                
                # Normalizacja punktów
                points -= points.min(axis=0)
                
                # Tworzenie chmury punktów
                point_cloud = pv.PolyData(points)
                point_cloud['Altitude'] = points[:, 2]

                # Tworzenie początkowej siatki
                surf = point_cloud.delaunay_2d()
                
                # Wygładzanie powierzchni w zależności od wybranej metody
                if smoothing_method == 'laplacian':
                    smoothed = surf.smooth(
                        n_iter=iterations,
                        relaxation_factor=relaxation_factor,
                        feature_angle=60,
                        boundary_smoothing=True
                    )
                elif smoothing_method == 'gaussian':
                    # Implementacja wygładzania gaussowskiego
                    smoothed = surf.copy()
                    points = smoothed.points
                    sigma = relaxation_factor  # Używamy współczynnika jako sigma
                    
                    for _ in range(iterations):
                        # Stwórz drzewo KD dla szybkiego wyszukiwania sąsiadów
                        tree = KDTree(points[:, :2])
                        new_points = points.copy()
                        
                        # Promień wyszukiwania (3 * sigma dla pokrycia 99.7% rozkładu Gaussa)
                        radius = 3 * sigma
                        
                        for i in range(len(points)):
                            # Znajdź sąsiadów w promieniu
                            neighbors = tree.query_ball_point(points[i, :2], r=radius)
                            if neighbors:
                                # Oblicz wagi gaussowskie
                                distances = np.sum((points[neighbors, :2] - points[i, :2])**2, axis=1)
                                weights = np.exp(-distances / (2 * sigma**2))
                                
                                # Normalizuj wagi
                                weights = weights / np.sum(weights)
                                
                                # Oblicz średnią ważoną dla wysokości
                                new_points[i, 2] = np.sum(weights * points[neighbors, 2])
                        
                        points = new_points
                        smoothed.points = points
                        
                elif smoothing_method == 'window':
                    smoothed = surf.copy()
                    points = smoothed.points
                    
                    for _ in range(iterations):
                        tree = KDTree(points[:, :2])
                        radius = 2.0 * relaxation_factor  # Promień zależny od współczynnika wygładzania
                        
                        for i in range(len(points)):
                            neighbors = tree.query_ball_point(points[i, :2], r=radius)
                            if neighbors:
                                weights = np.exp(-np.sum((points[neighbors, :2] - points[i, :2])**2, axis=1))
                                smoothed.points[i, 2] = np.average(points[neighbors, 2], weights=weights)

                # Wizualizacja
                self.frame.clear()

                # Dodaj wygładzoną powierzchnię
                self.frame.add_mesh(
                    smoothed,
                    scalars='Altitude',
                    cmap='terrain',
                    show_edges=False,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=10,
                    diffuse=0.7,
                    ambient=0.3
                )

                # Dodaj oryginalne punkty
                self.frame.add_points(
                    point_cloud,
                    color='red',
                    point_size=10,
                    render_points_as_spheres=True,
                    opacity=0.8
                )

                # Dodaj siatkę pomocniczą
                self.frame.show_grid(
                    grid=True,
                    location='outer',
                    all_edges=True,
                    xtitle='X [m]',
                    ytitle='Y [m]',
                    ztitle='Z [m]'
                )

                # Skala kolorów
                self.frame.add_scalar_bar(
                    title='Altitude [m]',
                    vertical=True,
                    position_x=0.85,
                    position_y=0.15,
                    n_labels=5,
                    fmt='%.1f',
                    width=0.1
                )

                # Statystyki
                stats = {
                    'Points': len(points),
                    'Min altitude': points[:, 2].min(),
                    'Max altitude': points[:, 2].max(),
                    'Range': points[:, 2].max() - points[:, 2].min(),
                    'Smoothing': smoothing_method,
                    'Iterations': iterations,
                    'Factor': relaxation_factor
                }
                
                stats_text = '\n'.join([
                    f"{k}: {v:.2f}m" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in stats.items()
                ])
                self.frame.add_text(
                    stats_text,
                    position='upper_left',
                    font_size=11,
                    shadow=False
                )

                # Kamera
                # self.frame.camera_position = 'iso'
                # self.frame.reset_camera()
                
                # Aktualizuj pasek statusu
                self.statusBar.showMessage(f"Model wygenerowany: {smoothing_method}, {iterations} iteracji, współczynnik {relaxation_factor}")

            except Exception as e:
                print(f"Błąd podczas wizualizacji: {str(e)}")
                self.statusBar.showMessage(f"Błąd: {str(e)}")
                import traceback
                traceback.print_exc()
        
    def process_points(self, points, radius=1.0):
        """
        Przetwarza i uśrednia punkty w skupiskach.
        
        Args:
            points: numpy array z punktami [x, y, z]
            radius: promień (w metrach) w którym punkty będą uśredniane
        
        Returns:
            Przetworzona chmura punktów z uśrednionymi skupiskami
        """
        from sklearn.cluster import DBSCAN
        
        # Użyj tylko współrzędnych X i Y do klastrowania
        xy_points = points[:, :2]
        
        # Znajdź klastry punktów używając DBSCAN
        clustering = DBSCAN(eps=radius, min_samples=2).fit(xy_points)
        labels = clustering.labels_
        
        # Przygotuj listę na przetworzone punkty
        processed_points = []
        
        # Dla każdego klastra (włączając szum, oznaczony jako -1)
        for label in range(-1, labels.max() + 1):
            mask = (labels == label)
            cluster_points = points[mask]
            
            if label == -1 or len(cluster_points) == 1:
                # Punkty oznaczone jako szum lub pojedyncze zachowaj bez zmian
                processed_points.extend(cluster_points)
            else:
                # Uśrednij punkty w klastrze
                mean_point = cluster_points.mean(axis=0)
                processed_points.append(mean_point)
        
        return np.array(processed_points)

    def analyze_point_distribution(self, points):
        """
        Analizuje rozkład punktów i wyświetla statystyki.
        """
        distances = []
        for i in range(len(points)):
            if i > 0:
                dist = np.linalg.norm(points[i] - points[i-1])
                distances.append(dist)
        
        stats = {
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances)
        }
        
        print(f"Statystyki odległości między punktami:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}m")
        
        return stats

    def visualize_terrain(self):
        if self.df is not None:
            # Konwersja danych na typ numeryczny
            self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
            self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
            self.df['Altitude'] = pd.to_numeric(self.df['Altitude'], errors='coerce')
            
            # Usuń brakujące dane
            self.df.dropna(subset=['Latitude', 'Longitude', 'Altitude'], inplace=True)
            
            # Konwertuj współrzędne geograficzne na metryczne
            points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                            zip(self.df['Latitude'], self.df['Longitude'], self.df['Altitude'])])
            
            if len(points) == 0:
                print("Brak danych do wizualizacji")
                return
            
            # Analizuj rozkład punktów przed przetworzeniem
            print("\nStatystyki przed przetworzeniem:")
            self.analyze_point_distribution(points)
            
            # Przetwórz i uśrednij punkty
            processed_points = self.process_points(points, radius=1.0)  # radius w metrach
            
            print(f"\nLiczba punktów przed przetworzeniem: {len(points)}")
            print(f"Liczba punktów po przetworzeniu: {len(processed_points)}")
            
            # Analizuj rozkład punktów po przetworzeniu
            print("\nStatystyki po przetworzeniu:")
            self.analyze_point_distribution(processed_points)
            
            # Normalizuj punkty
            processed_points -= processed_points.min(axis=0)
            
            # Stwórz chmurę punktów
            point_cloud = pv.PolyData(processed_points)
            point_cloud['Altitude'] = processed_points[:, 2]
            
            # Stwórz siatkę
            grid = point_cloud.delaunay_2d()
            
            # Wizualizacja
            self.frame.clear()
            
            # Dodaj siatkę
            self.frame.add_mesh(grid, 
                            scalars='Altitude', 
                            cmap='terrain',
                            show_edges=True,
                            edge_color='black',
                            line_width=1,
                            opacity=1.0)
            
            # Dodaj oryginalne punkty
            self.frame.add_points(point_cloud, 
                                color='blue', 
                                point_size=10,
                                render_points_as_spheres=True,
                                label='Measurement points',
                                opacity=1.0)
                
            # Dodaj elementy pomocnicze
            self.frame.show_grid()
            self.frame.show_axes()
            
            # Reset widoku kamery
            # self.frame.reset_camera()
            
            # Dodaj informacje o danych
            stats_text = (
                f"Points: {len(processed_points)}\n"
                f"Min altitude: {processed_points[:, 2].min():.2f}m\n"
                f"Max altitude: {processed_points[:, 2].max():.2f}m\n"
                f"Range: {processed_points[:, 2].max() - processed_points[:, 2].min():.2f}m"
            )
            self.frame.add_text(stats_text, position='upper_left', font_size=11)
            
        else:
            print("Brak danych do wizualizacji")
            
    def visualize_terrain_raw(self):
        if self.df is not None:
            # Konwersja danych na typ numeryczny
            self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
            self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
            self.df['Altitude'] = pd.to_numeric(self.df['Altitude'], errors='coerce')
            
            # Usuń brakujące dane
            self.df.dropna(subset=['Latitude', 'Longitude', 'Altitude'], inplace=True)
            
            # Konwertuj współrzędne geograficzne na metryczne
            points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                            zip(self.df['Latitude'], self.df['Longitude'], self.df['Altitude'])])
            
            if len(points) == 0:
                print("Brak danych do wizualizacji")
                return
            
            # Analizuj rozkład punktów przed przetworzeniem
            print("\nStatystyki przed przetworzeniem:")
            self.analyze_point_distribution(points)
            
            # Normalizuj punkty
            points -= points.min(axis=0)
            
            # Stwórz chmurę punktów
            point_cloud = pv.PolyData(points)
            point_cloud['Altitude'] = points[:, 2]
            
            # Stwórz siatkę
            grid = point_cloud.delaunay_2d()
            
            # Wizualizacja
            self.frame.clear()
            
            # Dodaj siatkę
            self.frame.add_mesh(grid, 
                            scalars='Altitude', 
                            cmap='terrain',
                            show_edges=True,
                            edge_color='black',
                            line_width=1)
            
            # Dodaj oryginalne punkty
            self.frame.add_points(point_cloud, 
                                color='darkred', 
                                point_size=5,
                                render_points_as_spheres=True,
                                label='Measurement points')
            
            # Dodaj elementy pomocnicze
            self.frame.show_grid()
            self.frame.show_axes()
            
            # Reset widoku kamery
            # self.frame.reset_camera()
            
            # Dodaj informacje o danych
            stats_text = (
                f"Points: {len(points)}\n"
                f"Min altitude: {points[:, 2].min():.2f}m\n"
                f"Max altitude: {points[:, 2].max():.2f}m\n"
                f"Range: {points[:, 2].max() - points[:, 2].min():.2f}m"
            )
            self.frame.add_text(stats_text, position='upper_left', font_size=11)
            
        else:
            print("Brak danych do wizualizacji")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TerrainMapper()
    window.show()
    sys.exit(app.exec())
