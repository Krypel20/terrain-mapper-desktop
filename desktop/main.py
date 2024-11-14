import sys
import pandas as pd
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from pyproj import Transformer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QComboBox, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QStatusBar, QToolBar, QMenuBar, QMenu, QStyle, QFrame, QFileDialog,
    QListWidget, QListWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
from pyvistaqt import QtInteractor
from db_connection import DatabaseConnection

class SessionListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
    def addSessionItem(self, session_name, measurements_count):
        item = QListWidgetItem(f"{session_name} ({measurements_count} points)")
        item.setData(Qt.ItemDataRole.UserRole, session_name)
        self.addItem(item)

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
            port="5434" #5433 // 5434 - Desktop // Laptop
        )
        # słownik do przechowywania ładowanych danych sesji
        self.session_data = {}

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
        
        # Dodanie usuwania sesji
        delete_action = QAction("Delete Sessions", self)
        delete_action.setShortcut("Del")
        delete_action.triggered.connect(self.delete_selected_sessions)
        
        file_menu.addAction(delete_action)
        file_menu.addSeparator()
        
        # Akcje w menu
        export_action = QAction("Export Model", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_model)
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
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
        
        # Session selection group
        session_group = QGroupBox("Session Selection")
        session_layout = QVBoxLayout()
        
        # Add session list widget
        self.session_list = SessionListWidget()
        self.session_list.itemSelectionChanged.connect(self.update_visualization_buttons)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Add refresh button
        self.load_button = QPushButton("Refresh Sessions")
        self.load_button.clicked.connect(self.load_sessions)
        
        # Add delete button
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_selected_sessions)
        self.delete_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                padding: 5px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #ff0000;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.delete_button.setEnabled(False)
        
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.delete_button)
        
        session_layout.addWidget(self.session_list)
        session_layout.addLayout(buttons_layout)
        session_group.setLayout(session_layout)
        
        # Visualization parameters group
        visualization_group = QGroupBox("Visualization Parameters")
        visualization_layout = QVBoxLayout()
        
        # Add merge options
        merge_label = QLabel("Merge Method:")
        self.merge_method = QComboBox()
        self.merge_method.addItems(['Simple', 'Average', 'Weighted'])
        
        # Add other parameters (smoothing, etc.)
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
        
        visualization_layout.addWidget(merge_label)
        visualization_layout.addWidget(self.merge_method)
        visualization_layout.addWidget(smoothing_label)
        visualization_layout.addWidget(self.smoothing_method)
        visualization_layout.addWidget(iterations_label)
        visualization_layout.addWidget(self.smoothing_iterations)
        visualization_layout.addWidget(factor_label)
        visualization_layout.addWidget(self.smoothing_factor)
        visualization_group.setLayout(visualization_layout)
        
        # Visualization buttons group
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
        
        # Add all groups to main layout
        control_layout.addWidget(session_group)
        control_layout.addWidget(visualization_group)
        control_layout.addWidget(buttons_group)
        control_layout.addStretch()
        
        # Styling
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
        """Load sessions from database and populate the session list."""
        self.session_list.clear()
        self.session_data.clear()
        
        sessions = self.db.get_sessions()
        for session_name in sessions:
            # Get the data for this session
            df = self.db.get_session_data(session_name)
            measurements_count = len(df) if df is not None else 0
            
            # Add to session list widget
            self.session_list.addSessionItem(session_name, measurements_count)
            
            # Store the data
            self.session_data[session_name] = df

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

    def create_visualization(self, points, raw=False):
        """Create visualization with real-world dimensions."""
        try:
            # Oblicz wymiary obszaru
            x_min, y_min = points[:, 0].min(), points[:, 1].min()
            x_max, y_max = points[:, 0].max(), points[:, 1].max()
            
            # Konwertuj współrzędne na względne od początku układu (0,0),
            # ale zachowaj rzeczywiste odległości
            normalized_points = points.copy()
            normalized_points[:, 0] -= x_min  # X
            normalized_points[:, 1] -= y_min  # Y
            # Nie normalizujemy Z (wysokości)
            
            # Oblicz wymiary terenu
            width = x_max - x_min
            length = y_max - y_min
            height = points[:, 2].max() - points[:, 2].min()
            
            # Tworzenie chmury punktów
            point_cloud = pv.PolyData(normalized_points)
            point_cloud['Altitude'] = points[:, 2]  # Oryginalne wysokości

            # Tworzenie siatki
            grid = point_cloud.delaunay_2d()

            # Generowanie iso-linii na siatce
            n_contours = 15  # liczba linii poziomicowych
            contour_values = np.linspace(
                points[:, 2].min(), 
                points[:, 2].max(), 
                n_contours
            )
            contours = grid.contour(contour_values)
            
            # Wizualizacja
            self.frame.clear()
            
            # Dodaj siatkę
            self.frame.add_mesh(grid, 
                            scalars='Altitude', 
                            cmap="terrain", 
                            show_edges=True if raw else False,
                            edge_color='black',
                            line_width=1)
            
            if not raw:
                # Dodaj iso-linie
                self.frame.add_mesh(
                    contours, 
                    color="black", 
                    line_width=2,
                    render_lines_as_tubes=True,
                    opacity=0.7
                )
            
            # Dodaj punkty
            self.frame.add_points(point_cloud, 
                                color='darkred' if raw else 'darkblue', 
                                point_size=5 if raw else 5,
                                render_points_as_spheres=True)
            
            # Dodaj siatkę pomocniczą z rzeczywistymi wymiarami
            self.frame.show_grid(
                grid=True,
                location='outer',
                all_edges=True,
                xtitle='X [m]',
                ytitle='Y [m]',
                ztitle='Z [m]'
            )
            self.frame.show_axes()
            
            # Statystyki z rzeczywistymi wymiarami
            stats = {
                'Points': len(points),
                'Area width': f"{width:.1f}m",
                'Area length': f"{length:.1f}m",
                'Min altitude': f"{float(points[:, 2].min()):.2f}m",
                'Max altitude': f"{float(points[:, 2].max()):.2f}m",
                'height difference': f"{height:.2f}m",
                'Area': f"{(width * length):.1f}m²",
                'Type': 'Raw' if raw else 'Processed'
            }
            
            stats_text = '\n'.join([
                f"{k}: {v}" for k, v in stats.items()
            ])
            self.frame.add_text(
                stats_text,
                position='upper_left',
                font_size=11,
                shadow=False
            )
            
            self.statusBar.showMessage(f"Visualization created: {stats['Type']} point cloud")
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            self.statusBar.showMessage(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_smooth_surface(self, points, smoothing_method, iterations, relaxation_factor):
        """Create smoothed terrain surface visualization with corrected texturing."""
        try:
            print(f"Applying smoothing parameters:")
            print(f"Method: {smoothing_method}")
            print(f"Iterations: {iterations}")
            print(f"Smoothing factor: {relaxation_factor}")

            # Oblicz wymiary obszaru
            x_min, y_min = points[:, 0].min(), points[:, 1].min()
            x_max, y_max = points[:, 0].max(), points[:, 1].max()
            
            # Konwertuj współrzędne na względne od początku układu (0,0),
            # ale zachowaj rzeczywiste odległości
            normalized_points = points.copy()
            normalized_points[:, 0] -= x_min  # X
            normalized_points[:, 1] -= y_min  # Y
            # Nie normalizujemy Z (wysokości)
            
            # Oblicz wymiary terenu
            width = x_max - x_min
            length = y_max - y_min
            height = points[:, 2].max() - points[:, 2].min()
            
            # Tworzenie chmury punktów
            point_cloud = pv.PolyData(normalized_points)  # Użyj znormalizowanych punktów
            point_cloud['Altitude'] = points[:, 2]  # Oryginalne wysokości

            # Tworzenie początkowej siatki
            surf = point_cloud.delaunay_2d()
            
            # Zainicjuj zmienną smoothed przed warunkami
            smoothed = None
            
            # Wygładzanie powierzchni
            if smoothing_method == 'laplacian':
                smoothed = surf.smooth(
                    n_iter=iterations,
                    relaxation_factor=relaxation_factor,
                    feature_angle=60,
                    boundary_smoothing=True,
                    convergence=0.1,
                    progress_bar=True
                )
            elif smoothing_method == 'gaussian':
                smoothed = surf.copy()
                pts = smoothed.points
                sigma = relaxation_factor
                
                for _ in range(iterations):
                    tree = KDTree(pts[:, :2])
                    new_points = pts.copy()
                    radius = 3 * sigma
                    
                    for i in range(len(pts)):
                        neighbors = tree.query_ball_point(pts[i, :2], r=radius)
                        if neighbors:
                            distances = np.sum((pts[neighbors, :2] - pts[i, :2])**2, axis=1)
                            weights = np.exp(-distances / (2 * sigma**2))
                            weights = weights / np.sum(weights)
                            new_points[i, 2] = np.sum(weights * pts[neighbors, 2])
                    
                    pts = new_points
                    smoothed.points = pts
                    
            elif smoothing_method == 'window':
                smoothed = surf.copy()
                pts = smoothed.points
                
                for _ in range(iterations):
                    tree = KDTree(pts[:, :2])
                    radius = 2.0 * relaxation_factor
                    
                    for i in range(len(pts)):
                        neighbors = tree.query_ball_point(pts[i, :2], r=radius)
                        if neighbors:
                            weights = np.exp(-np.sum((pts[neighbors, :2] - pts[i, :2])**2, axis=1))
                            smoothed.points[i, 2] = np.average(pts[neighbors, 2], weights=weights)

            # Sprawdź czy wygładzanie się powiodło
            if smoothed is None:
                raise ValueError(f"Smoothing failed with method: {smoothing_method}")

            # Pobierz punkty po wygładzeniu
            smoothed_points = smoothed.points
            
            # Aktualizacja wartości wysokości po wygładzeniu
            smoothed.point_data['Altitude'] = smoothed_points[:, 2]
            
            # Obliczenie nachylenia terenu
            terrain_slope = smoothed.compute_derivative(scalars='Altitude')
            
            # Obliczanie nachylenie terenu modelu w procentach
            # Obliczenie kąta w radianach używając arctangensa gradientu
            # Konwersja na procenty: tan(kąt) * 100
            gradient_x = terrain_slope['gradient'][:, 0]
            gradient_y = terrain_slope['gradient'][:, 1]
            slope_magnitude = np.sqrt(gradient_x**2 + gradient_y**2) * 100  # konwersja na procenty
            smoothed.point_data['Slope'] = slope_magnitude

            # Wizualizacja
            self.frame.clear()

            # Dodaj wygładzoną powierzchnię z zaktualizowaną teksturą
            mesh_actor = self.frame.add_mesh(
                smoothed,
                scalars='Altitude',
                cmap='terrain',
                show_edges=False,
                smooth_shading=True,
                specular=0.3,
                specular_power=10,
                diffuse=0.7,
                ambient=0.3,
                scalar_bar_args={
                    'title': 'Altitude [m]',
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.15,
                    'n_labels': 5,
                    'fmt': '%.1f',
                    'width': 0.1
                }
            )

            # Dodaj punkty
            self.frame.add_points(point_cloud, 
                                color='lightblue', 
                                point_size=2,
                                render_points_as_spheres=True)

            # Generowanie iso-linii na wygładzonym modelu
            n_contours = 20
            contour_values = np.linspace(
                smoothed_points[:, 2].min(),
                smoothed_points[:, 2].max(),
                n_contours
            )
            contours = smoothed.contour(contour_values)
        
            # Dodaj iso-linie z dostosowaniem do aktualnego skalara
            self.frame.add_mesh(
                contours, 
                color="black", 
                line_width=2,
                render_lines_as_tubes=True,
                opacity=0.7,
                name='contours'  # nazwa do późniejszego usunięcia
            )

            # Siatka pomocnicza
            self.frame.show_grid(
                grid=True,
                location='outer',
                all_edges=True,
                xtitle='X [m]',
                ytitle='Y [m]',
                ztitle='Z [m]'
            )
            
            # Referencje do oryginalnych kolormap
            self.altitude_cmap = 'terrain'
            self.slope_cmap = 'fire'
            
            # Zmienna do śledzenia aktualnego skalara
            self.current_scalar = 'Altitude'

            # Poprawiona funkcja przełączająca skalar
            def toggle_scalar(flag):
                try:
                    # Usuń stare iso-linie
                    self.frame.remove_actor('contours')

                    if self.current_scalar == 'Altitude':
                        # Przełącz na nachylenie
                        mesh_actor.mapper.array_name = 'Slope'
                        mesh_actor.mapper.scalar_range = [slope_magnitude.min(), slope_magnitude.max()]
                        mesh_actor.mapper.lookup_table.cmap = 'viridis'
                        self.frame.scalar_bar.SetTitle('Slope [%]')
                        self.current_scalar = 'Slope'

                        # Generuj iso-linie dla nachylenia
                        contours = smoothed.contour(isovalues=np.linspace(
                            smoothed_points[:, 2].min(),
                            smoothed_points[:, 2].max(),
                            20
                        ))
                        self.frame.add_mesh(
                            contours, 
                            color="white", 
                            line_width=2,
                            render_lines_as_tubes=True,
                            opacity=0.7,
                            name='contours'  # nazwa do późniejszego usunięcia
                        )
                    else:
                        # Przełącz na wysokość
                        mesh_actor.mapper.array_name = 'Altitude'
                        mesh_actor.mapper.scalar_range = [smoothed_points[:, 2].min(), smoothed_points[:, 2].max()]
                        mesh_actor.mapper.lookup_table.cmap = 'magma'
                        self.frame.scalar_bar.SetTitle('Altitude [m]')
                        self.current_scalar = 'Altitude'

                        # Generuj iso-linie dla wysokości
                        slope_contour_values = np.linspace(
                            slope_magnitude.min(),
                            slope_magnitude.max(),
                            15
                        )
                        slope_contours = smoothed.contour(
                            slope_contour_values,
                            scalars='Slope'
                        )
                        self.frame.add_mesh(
                            slope_contours,
                            color="black",
                            line_width=2,
                            render_lines_as_tubes=True,
                            opacity=0.5,
                            name='contours'
                        )

                    # Aktualizacja widoku
                    self.frame.render()
                    
                    # Aktualizuj pasek statusu
                    self.statusBar.showMessage(f"Displaying: {self.current_scalar}")
                    
                except Exception as e:
                    print(f"Error in toggle_scalar: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Dodaj przycisk przełączający z etykietą
            self.frame.add_checkbox_button_widget(
                toggle_scalar,
                value=False,
                size=30,
                position=(10, 10),
                color_on='green',
                color_off='red',
                background_color='black'
            )

            # Dodaj tekst opisujący przycisk
            self.frame.add_text(
                'Toggle Height/Slope',
                position=(50, 10),
                font_size=9,
                color='black',
                shadow=True
            )

            # Przygotuj statystyki
            stats = {
                'Points': len(smoothed_points),
                'Area width': f"{width:.1f}m",
                'Area length': f"{length:.1f}m",
                'Min altitude': f"{float(points[:, 2].min()):.2f}m",
                'Max altitude': f"{float(points[:, 2].max()):.2f}m",
                'Height range': f"{height:.2f}m",
                'Area': f"{(width * length):.1f}m²",
                'Max slope': f"{float(slope_magnitude.max()):.1f}%",
                'Mean slope': f"{float(slope_magnitude.mean()):.1f}%",
                'Smoothing': smoothing_method,
                'Iterations': iterations,
                'Factor': relaxation_factor
            }
            
            # Zaktualizowany format wyświetlania statystyk
            stats_text = '\n'.join([
                f"{k}: {v}m" if k in ['Min altitude', 'Max altitude', 'Range'] else
                f"{k}: {v}" for k, v in stats.items()
            ])
            
            self.frame.add_text(
                stats_text,
                position='upper_left',
                font_size=11,
                shadow=False
            )
            
            self.statusBar.showMessage(
                f"Model generated: {smoothing_method}, {iterations} iterations, factor {relaxation_factor}"
            )

        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            self.statusBar.showMessage(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def visualize_terrain_interpolation(self):
        """Visualize interpolated surface from merged sessions."""
        if self.df is None:
            self.df = self.merge_session_data()
            
        if self.df is None:
            self.statusBar.showMessage("No data to visualize")
            return
        
        # Get visualization parameters
        smoothing_method = self.smoothing_method.currentText()
        iterations = self.smoothing_iterations.value()
        relaxation_factor = self.smoothing_factor.value()
        
        # Konwersja kolumny Altitude na float
        self.df['Altitude'] = self.df['Altitude'].astype(float)
        
        # Convert to metric coordinates
        points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                        zip(self.df['Latitude'], self.df['Longitude'], self.df['Altitude'])])
        
        # Process points
        processed_points = self.process_points(points, radius=1.0)
        
        # Create and smooth surface
        self.create_smooth_surface(processed_points, smoothing_method, iterations, relaxation_factor)

    def visualize_terrain_raw(self):
        """Visualize raw point cloud from merged sessions."""
        merged_df = self.merge_session_data()
        if merged_df is None:
            self.statusBar.showMessage("No data to visualize")
            return
            
        self.df = merged_df  # Store for other visualizations
        
        # Konwersja kolumny Altitude na float
        self.df['Altitude'] = self.df['Altitude'].astype(float)
        
        # Convert to metric coordinates
        points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                        zip(merged_df['Latitude'], merged_df['Longitude'], merged_df['Altitude'])])
        
        # Create visualization
        self.create_visualization(points, raw=True)

    def visualize_terrain(self):
        """Visualize processed point cloud from merged sessions."""
        if self.df is None:
            self.df = self.merge_session_data()
            
        if self.df is None:
            self.statusBar.showMessage("No data to visualize")
            return
        
        # Konwersja kolumny Altitude na float
        self.df['Altitude'] = self.df['Altitude'].astype(float)
            
        points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                        zip(self.df['Latitude'], self.df['Longitude'], self.df['Altitude'])])
        
        # Process points
        processed_points = self.process_points(points, radius=1.0)
        
        # Create visualization
        self.create_visualization(processed_points)

    def update_visualization_buttons(self):
        """Enable/disable buttons based on selection."""
        selected_items = self.session_list.selectedItems()
        enable_buttons = len(selected_items) > 0
        
        self.visualize_button_raw.setEnabled(enable_buttons)
        self.visualize_button.setEnabled(enable_buttons)
        self.visualize_button_interpolation.setEnabled(enable_buttons)
        self.delete_button.setEnabled(enable_buttons)

    def delete_selected_sessions(self):
        """Delete selected sessions after confirmation."""
        selected_items = self.session_list.selectedItems()
        if not selected_items:
            return
            
        session_names = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        
        # Pokaż dialog potwierdzenia
        sessions_str = "\n".join(session_names)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText("Are you sure you want to delete the following sessions?")
        msg.setInformativeText(f"Selected sessions:\n{sessions_str}\n\nThis action cannot be undone!")
        msg.setWindowTitle("Confirm Deletion")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        
        if msg.exec() == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            for session_name in session_names:
                if self.db.delete_session(session_name):
                    deleted_count += 1
            
            # Odśwież listę sesji
            self.load_sessions()
            
            # Wyczyść aktualną wizualizację
            if hasattr(self, 'frame'):
                self.frame.clear()
            
            # Aktualizuj status
            self.statusBar.showMessage(f"Successfully deleted {deleted_count} session(s)")
            
            # Jeśli usunięto aktualnie wyświetlaną sesję, wyczyść dane
            if self.df is not None and any(session_name in self.df['session_name'].values for session_name in session_names):
                self.df = None

    def merge_session_data(self):
        """Merge selected session data based on chosen method."""
        selected_items = self.session_list.selectedItems()
        if not selected_items:
            return None
            
        selected_sessions = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        merge_method = self.merge_method.currentText()
        
        all_data = []
        for session_name in selected_sessions:
            df = self.session_data.get(session_name)
            if df is not None:
                df['session_name'] = session_name  # Add session identifier
                all_data.append(df)
        
        if not all_data:
            return None
            
        merged_df = pd.concat(all_data, ignore_index=True)
        
        if merge_method == 'Average':
            # Group by close points and average their values
            merged_df = self.average_nearby_points(merged_df)
        elif merge_method == 'Weighted':
            # Apply weighted averaging based on measurement quality
            merged_df = self.weighted_average_points(merged_df)
            
        return merged_df
    
    def average_nearby_points(self, df, radius=1.0):
        """Average points that are within specified radius of each other."""
        points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                          zip(df['Latitude'], df['Longitude'], df['Altitude'])])
        
        tree = KDTree(points[:, :2])
        processed_points = []
        processed_indices = set()
        
        for i in range(len(points)):
            if i in processed_indices:
                continue
                
            # Find nearby points
            nearby_indices = tree.query_ball_point(points[i, :2], r=radius)
            if nearby_indices:
                # Average the points
                cluster_points = points[nearby_indices]
                mean_point = cluster_points.mean(axis=0)
                processed_points.append(mean_point)
                processed_indices.update(nearby_indices)
        
        result_df = pd.DataFrame({
            'Latitude': [p[0] for p in processed_points],
            'Longitude': [p[1] for p in processed_points],
            'Altitude': [p[2] for p in processed_points]
        })
        
        return result_df
    
    def weighted_average_points(self, df, radius=1.0):
        """Average points with weights based on measurement quality."""
        points = np.array([self.geo_to_metric(lat, lon, alt) for lat, lon, alt in 
                          zip(df['Latitude'], df['Longitude'], df['Altitude'])])
        
        # Create weights based on available quality metrics
        if 'HDOP' in df.columns:
            # Lower HDOP means better quality
            weights = 1 / df['HDOP']
        else:
            weights = np.ones(len(points))
        
        tree = KDTree(points[:, :2])
        processed_points = []
        processed_indices = set()
        
        for i in range(len(points)):
            if i in processed_indices:
                continue
                
            nearby_indices = tree.query_ball_point(points[i, :2], r=radius)
            if nearby_indices:
                cluster_points = points[nearby_indices]
                cluster_weights = weights[nearby_indices]
                
                # Calculate weighted average
                weighted_mean = np.average(cluster_points, weights=cluster_weights, axis=0)
                processed_points.append(weighted_mean)
                processed_indices.update(nearby_indices)
        
        result_df = pd.DataFrame({
            'Latitude': [p[0] for p in processed_points],
            'Longitude': [p[1] for p in processed_points],
            'Altitude': [p[2] for p in processed_points]
        })
        
        return result_df

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TerrainMapper()
    window.show()
    sys.exit(app.exec())
