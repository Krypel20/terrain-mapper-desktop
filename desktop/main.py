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
from PyQt6.QtGui import QAction, QColor, QIcon
from pyvistaqt import QtInteractor
from db_connection import DatabaseConnection
from pykrige.ok import OrdinaryKriging
from geopy.geocoders import Nominatim
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QProgressDialog

class LoadSessionsThread(QThread):
    finished = pyqtSignal(object)
    
    def __init__(self, db, view_mode):
        super().__init__()
        self.db = db
        self.view_mode = view_mode
        
    def run(self):
        try:
            if self.view_mode == 'By Date':
                # Pobierz prostą listę sesji
                sessions = self.db.get_sessions()
                self.finished.emit({'mode': 'date', 'data': sessions})
            else:
                # Pobierz pogrupowane sesje
                groups = self.db.get_session_groups()
                self.finished.emit({'mode': 'location', 'data': groups})
        except Exception as e:
            print(f"Error loading sessions: {e}")
            self.finished.emit(None)
            
class SessionListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
    def addSessionItem(self, session_name, measurements_count):
        item = QListWidgetItem(f"{session_name} ({measurements_count} points)")
        item.setData(Qt.ItemDataRole.UserRole, session_name)
        # Nie ustawiamy tu żadnych kolorów - zostawiamy domyślne
        self.addItem(item)
        return item

    def addSeparator(self, group_name):
        separator = QListWidgetItem(group_name)
        separator.setFlags(separator.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        separator.setBackground(QColor(80, 80, 80))  # Ciemniejszy szary dla separatora
        font = separator.font()
        font.setBold(True)
        separator.setFont(font)
        self.addItem(separator)

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
            port="5433" #5433 // 5434 - Desktop // Laptop
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
        self.points_actor = None
        self.normalized_cloud = None
        
        # Załaduj sesje
        self.load_sessions()
        
    def toggle_points(self, flag):
        """Metoda do przełączania widoczności punktów pomiarowych."""
        if flag and self.normalized_cloud is not None:
            # Dodaj punkty używając znormalizowanej chmury
            self._points_actor = self.frame.add_mesh(
                self.normalized_cloud,
                color='crimson',
                point_size=4,
                render_points_as_spheres=False,
            )
        else:
            # Usuń punkty jeśli istnieją
            if self._points_actor is not None:
                self.frame.remove_actor(self._points_actor)
                self._points_actor = None
        self.frame.render()

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
        
        # Przyciski odświerzania
        refresh_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Refresh", self)
        refresh_action.triggered.connect(self.load_sessions)
        toolbar.addAction(refresh_action)
        
        toolbar.addSeparator()
        
        # Przycisk eksportu
        export_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Export Model", self)
        export_action.triggered.connect(self.export_model)
        toolbar.addAction(export_action)

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
        
        # Dodaj przełącznik widoku przed listą sesji
        view_layout = QHBoxLayout()
        view_label = QLabel("View Mode:")
        self.view_mode = QComboBox()
        self.view_mode.addItems(['By Date', 'By Location'])
        self.view_mode.currentTextChanged.connect(self.load_sessions)
        
        view_layout.addWidget(view_label)
        view_layout.addWidget(self.view_mode)
        
        session_layout.insertLayout(0, view_layout)  # Dodaj na początku layoutu sesji
        
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
        self.smoothing_method.addItems(['laplacian', 'gaussian'])
        
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
        
        self.visualize_button_kriging = QPushButton("Create Kriging Model")
        self.visualize_button_kriging.setEnabled(False)
        self.visualize_button_kriging.setStyleSheet("""
            QPushButton {
                background-color: #2E8B57;
                color: white;
                padding: 5px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3CB371;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        buttons_layout.addWidget(self.visualize_button_raw)
        buttons_layout.addWidget(self.visualize_button)
        buttons_layout.addWidget(self.visualize_button_interpolation)
       #buttons_layout.addWidget(self.visualize_button_kriging)
        buttons_group.setLayout(buttons_layout)
        
        # Add all groups to main layout
        control_layout.addWidget(session_group)
        control_layout.addWidget(visualization_group)
        control_layout.addWidget(buttons_group)
        control_layout.addStretch()
        
        # Styling
        control_widget.setMaximumWidth(350)
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
        """Eksportuje aktualnie wyświetlany model do wybranego formatu."""
        if not hasattr(self, 'grid') or self.grid is None:
            self.statusBar.showMessage("No model to export. Create a visualization first.")
            return
            
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Model",
            "",
            "STL Files (*.stl);;VTK Files (*.vtk);;OBJ Files (*.obj)"
        )
        
        if not file_name:
            return
            
        try:
            # Eksport w zależności od formatu
            if file_name.endswith('.stl'):
                self.grid.save(file_name, binary=True)
            elif file_name.endswith('.vtk'):
                self.grid.save(file_name)
            elif file_name.endswith('.obj'):
                self.grid.save(file_name)
                
            self.statusBar.showMessage(f"Model successfully exported to {file_name}")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error exporting model: {str(e)}")
            print(f"Export error details: {str(e)}")

    def load_sessions(self):
        """Załadowanie sesji z bazy danych."""
        progress = QProgressDialog("Loading sessions...", None, 0, 0, self)
        progress.setWindowTitle("Please wait")
        progress.setModal(True)
        progress.setCancelButton(None)
        progress.show()
        
        # Utwórz i uruchom wątek ładowania
        self.load_thread = LoadSessionsThread(self.db, self.view_mode.currentText())
        self.load_thread.finished.connect(lambda data: self._update_session_list(data, progress))
        self.load_thread.start()
            
    def _update_session_list(self, data, progress):
        """Update session list based on selected view mode."""
        if not data:
            progress.close()
            return
            
        self.session_list.clear()
        self.session_data.clear()
        
        if data['mode'] == 'date':
            # Wyświetl sesje chronologicznie
            for session_name in data['data']:
                df = self.db.get_session_data(session_name)
                measurements_count = len(df) if df is not None else 0
                self.session_list.addSessionItem(session_name, measurements_count)
                self.session_data[session_name] = df
        else:
            # Wyświetl sesje pogrupowane według lokalizacji
            for group_index, group in enumerate(data['data']):
                if not group:
                    continue
                    
                if group_index > 0:
                    spacer = QListWidgetItem("")
                    spacer.setFlags(spacer.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                    self.session_list.addItem(spacer)
                
                group_coords = self._get_location_group_name(group[0])
                group_name = f"Group {group_index + 1} - {group_coords}"
                self.session_list.addSeparator(group_name)
                
                for session_name in group:
                    df = self.db.get_session_data(session_name)
                    measurements_count = len(df) if df is not None else 0
                    self.session_list.addSessionItem(session_name, measurements_count)
                    self.session_data[session_name] = df
        
        progress.close()

    def load_session_data(self):
        session_name = self.session_combo.currentText()
        if session_name:
            self.df = self.db.get_session_data(session_name)
            self.visualize_button_raw.setEnabled(True)
            self.visualize_button.setEnabled(True)
            self.visualize_button_interpolation.setEnabled(True)
            # self.visualize_button_kriging.setEnabled(True)

    def _get_location_group_name(self, session_name):
        """Generates the group name based on the session's location."""
        query = """
        SELECT 
            ST_Y(area_centroid::geometry) as lat,
            ST_X(area_centroid::geometry) as lon
        FROM measurement_sessions 
        WHERE session_name = %s
        """
        
        try:
            with self.db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, [session_name])
                    lat, lon = cur.fetchone()
                    
                    # Użyj geopy do uzyskania nazwy lokalizacji
                    geolocator = Nominatim(user_agent="terrain_mapper_app")
                    location = geolocator.reverse(f"{lat},{lon}")
                    address = location.raw['address']
                    town = address.get('town', '') or address.get('village', '') or address.get('city', '')
                    
                    # Zwróć nazwę lokalizacji wraz z koordynatami
                    return f"{town} ({lat:.3f}, {lon:.3f})"
        except Exception as e:
            print(f"Error getting location name: {e}")
            return f"{lat:.4f}°N, {lon:.4f}°E"  # Fallback do samych koordynatów

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
        
        return smoothed
 
    def process_points(self, points, radius=1.0):
        """
        Przetwarza punkty pomiarowe poprzez analizę skupisk i korektę wysokości.
        
        Args:
            points: tablica numpy z punktami [x, y, z]
            radius: promień w metrach do wyszukiwania skupisk w płaszczyźnie XY
        
        Returns:
            Przetworzona chmura punktów z skorygowanymi wysokościami
        """
        from sklearn.neighbors import BallTree
        import numpy as np
        from scipy import stats
        
        def analyze_cluster_heights(heights):
            """
            Analizuje wysokości w skupisku i usuwa wartości odstające.
            Zwraca listę poprawnych wysokości i informacje o odrzuconych punktach.
            """
            # Oblicz podstawowe statystyki
            q1 = np.percentile(heights, 25)
            q3 = np.percentile(heights, 75)
            iqr = q3 - q1
            median = np.median(heights)
            
            # Określ granice dla wartości odstających (1.5 * IQR)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Znajdź wartości mieszczące się w zakresie
            valid_mask = (heights >= lower_bound) & (heights <= upper_bound)
            valid_heights = heights[valid_mask]
            outliers = heights[~valid_mask]
            
            return valid_heights, outliers, median

        # Utwórz drzewo do szybkiego wyszukiwania sąsiadów w płaszczyźnie XY
        tree = BallTree(points[:, :2])
        processed_points = []
        processed_indices = set()
        
        # Przetwórz każdy punkt
        for i in range(len(points)):
            if i in processed_indices:
                continue
            
            # Znajdź wszystkie punkty w zadanym promieniu (tylko XY)
            nearby_indices = tree.query_radius([points[i, :2]], r=radius)[0]
            
            if len(nearby_indices) > 0:
                cluster_points = points[nearby_indices]
                heights = cluster_points[:, 2]
                
                # Jeśli w skupisku jest więcej niż jeden punkt
                if len(heights) > 1:
                    # Analizuj wysokości i usuń wartości odstające
                    valid_heights, outliers, median = analyze_cluster_heights(heights)
                    
                    # Jeśli znaleziono wartości odstające, wyświetl informacje
                    if len(outliers) > 0:
                        print(f"\nZnaleziono wartości odstające w skupisku:")
                        print(f"Pozycja środka XY: ({cluster_points[:, 0].mean():.2f}, {cluster_points[:, 1].mean():.2f})")
                        print(f"Wszystkie wysokości: {sorted(heights)}")
                        print(f"Odrzucone wysokości: {sorted(outliers)}")
                        print(f"Zaakceptowane wysokości: {sorted(valid_heights)}")
                    
                    # Oblicz końcową wysokość jako średnią z poprawnych wartości
                    if len(valid_heights) > 0:
                        final_height = np.mean(valid_heights)
                    else:
                        # Jeśli wszystkie punkty zostały uznane za odstające, użyj mediany
                        final_height = median
                        print("Uwaga: wszystkie punkty w skupisku zostały uznane za odstające")
                        print(f"Użyto mediany: {median}")
                    
                    # Utwórz nowy punkt ze średnią pozycją XY i obliczoną wysokością
                    new_point = np.array([
                        cluster_points[:, 0].mean(),
                        cluster_points[:, 1].mean(),
                        final_height
                    ])
                    processed_points.append(new_point)
                    
                else:
                    # Jeśli w skupisku jest tylko jeden punkt, zachowaj go bez zmian
                    processed_points.append(cluster_points[0])
                
                # Oznacz przetworzone punkty
                processed_indices.update(nearby_indices)
        
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

            # Tworzenie siatki trójkątów
            grid = point_cloud.delaunay_2d()

            # Generowanie iso-linii na siatce
            n_contours = 10 # liczba linii poziomicowych
            contour_values = np.linspace(
                points[:, 2].min(), 
                points[:, 2].max(), 
                n_contours
            )
            contours = grid.contour(contour_values)
            
            # Wizualizacja
            self.frame.clear()
            
            # Przygotowanie siatki
            grid.compute_normals(inplace=True)  # Oblicz normalne przed dodaniem siatki
            
            # Dodaj siatkę
            self.frame.add_mesh(grid, 
                  scalars='Altitude', 
                  cmap="terrain", 
                  show_edges=True if raw else False,
                  edge_color='black',
                  line_width=1,
                  smooth_shading=True,
                  specular=0.5,
                  specular_power=15,
                  diffuse=0.8,
                  ambient=0.3,
                  opacity=1.0,
                  scalar_bar_args={
                      'title': 'Wysokość [m]',
                      'vertical': True,
                      'position_x': 0.85,
                      'position_y': 0.15,
                      'n_labels': 5,
                      'fmt': '%.1f',
                      'width': 0.1,
                      'font_family': 'arial',
                      'bold': True,
                      'shadow': True,
                      'color': 'black',
                      'background_color': 'white',
                  })
            
            if not raw:
                # Dodaj iso-linie
                # self.frame.add_mesh(
                #     contours, 
                #     color="black", 
                #     line_width=2,
                #     render_lines_as_tubes=True,
                #     opacity=0.7
                # )
                
                # Dodaj światła
                light1 = pv.Light(light_type='headlight')  # Światło z "przodu"
                light1.intensity = 0.8
                self.frame.add_light(light1)

                light2 = pv.Light(position=[1, 1, 1], focal_point=[0, 0, 0])  # Światło boczne
                light2.intensity = 0.6
                self.frame.add_light(light2)
            
            # Dodaj punkty
            self.frame.add_points(point_cloud, 
                                color='darkred' if raw else 'darkblue', 
                                point_size=5 if raw else 5,
                                render_points_as_spheres=False)
            
            # Oblicz zakres wartości Z
            z_min = int(points[:, 2].min())
            z_max = int(points[:, 2].max())

            # Dodaj siatkę pomocniczą z rzeczywistymi wymiarami
            self.frame.show_grid(
                grid=True,
                location='outer',
                all_edges=True,
                xtitle='X [m]',
                ytitle='Y [m]',
                ztitle='Z [m]',
                font_size=12,
                padding=0.2,
                fmt='%d',
                bounds=[-width/2, width/2,      # zakres X
                        -length/2, length/2,     # zakres Y
                        z_min, z_max]           # zakres Z
            )

            self.frame.show_axes()

            # Przygotuj i wyświetl statystyki
            stats = {
                'Points': len(points),
                'Width': f"{width:.1f}m",
                'Length': f"{length:.1f}m",
                'Min altitude': f"{int(points[:, 2].min())}m",
                'Max altitude': f"{int(points[:, 2].max())}m",
                'Height difference': f"{int(height)}m",
                'Area': f"{(width * length):.1f}m²",
                'Type': 'Raw' if raw else 'Processed'  # Dodajemy informację o typie
            }

            stats_text = '\n'.join([f"{k}: {v}" for k, v in stats.items()])
            self.frame.add_text(stats_text, position='upper_left', font_size=11, shadow=False)

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
                points = smoothed.points.copy()
                
                # Oblicz średnią odległość między punktami
                tree = KDTree(points[:, :2])
                # Znajdź 2 najbliższych sąsiadów dla każdego punktu (pierwszy to sam punkt)
                distances = []
                for i in range(min(100, len(points))):
                    dist = tree.query(points[i, :2].reshape(1, -1), k=2)[0]
                    if len(dist[0]) > 1:
                        distances.append(dist[0][1])
                
                mean_distance = np.mean(distances) if distances else 1.0
                sigma = mean_distance * relaxation_factor
                
                for _ in range(iterations):
                    new_points = points.copy()
                    search_radius = 3.0 * sigma
                    
                    for i in range(len(points)):
                        neighbors_idx = tree.query_ball_point(points[i, :2], r=search_radius)
                        
                        if len(neighbors_idx) > 1:
                            diff = points[neighbors_idx, :2] - points[i, :2]
                            dist_sq = np.sum(diff * diff, axis=1)
                            weights = np.exp(-0.5 * dist_sq / (sigma * sigma))
                            weights = weights / np.sum(weights)
                            new_points[i, 2] = np.sum(weights * points[neighbors_idx, 2])
                    
                    points = new_points.copy()
                
                smoothed.points = points

            # Sprawdź czy wygładzanie się powiodło
            if smoothed is None:
                raise ValueError(f"Smoothing failed with method: {smoothing_method}")

            # Pobierz punkty po wygładzeniu
            smoothed_points = smoothed.points
            
            # Aktualizacja wartości wysokości po wygładzeniu
            smoothed.point_data['Altitude'] = smoothed_points[:, 2]
            
            # Przygotuj statystyki
            stats = {
                'Punkty': len(smoothed_points),
                'Szerokość obszaru': f"{width:.1f}m",
                'Długość obszaru': f"{length:.1f}m",
                'Min wysokość': f"{float(points[:, 2].min()):.2f}m",
                'Max wysokość': f"{float(points[:, 2].max()):.2f}m",
                'Zmiana wysokości': f"{height:.2f}m",
                'Obszar': f"{(width * length):.1f}m²",
                #'Średnie nachylenie': f"{average_slope:.1f}%",
                'Metoda wygładaznia': smoothing_method,
                'Liczba iteracji': iterations,
                'Współczynnik': relaxation_factor
            }
            
            # Oblicz zakres wysokości dla głębokości bloku
            z_min = points[:, 2].min()
            
            # Wykonaj ekstruzję wygładzonej powierzchni
            extruded = smoothed.extrude([0, 0, -1], capping=True)
            
             # Pobierz liczbę punktów w oryginalnej powierzchni
            n_original = len(smoothed.points)
            
            # Ustaw wszystkie dolne punkty na tę samą wysokość z_min
            points_extruded = extruded.points
            for i in range(n_original, len(points_extruded)):
                points_extruded[i][2] = z_min
            
            extruded.points = points_extruded
            
            # Stwórz tablicę kolorowania tylko dla górnej powierzchni
            altitude_colors = np.ones(len(extruded.points)) * z_min
            altitude_colors[:n_original] = smoothed.point_data['Altitude']
            extruded.point_data['Altitude'] = altitude_colors

            # Zapisz siatkę do późniejszego eksportu
            self.grid = extruded
            
                        # Znajdź punkty na granicy górnej powierzchni
            boundary_edges = smoothed.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=False,
                feature_edges=False,
                manifold_edges=False
            )
            
            # Stwórz linie pionowe łączące górną i dolną powierzchnię
            boundary_points = boundary_edges.points
            n_points = len(boundary_points)

            # Przygotuj punkty i linie
            vertical_points = []
            vertical_lines = []
            point_id = 0
            
            for point in boundary_points:
                # Dodaj górny punkt
                vertical_points.append(point.copy())
                # Dodaj dolny punkt
                bottom_point = point.copy()
                bottom_point[2] = z_min
                vertical_points.append(bottom_point)
                
                # Dodaj definicję linii (2 punkty)
                vertical_lines.append([2, point_id, point_id + 1])
                point_id += 2
                
            # Stwórz obiekt PolyData z poprawną strukturą linii
            vertical_edges = pv.PolyData(
                np.array(vertical_points),
                np.array(vertical_lines)
            )

            # Wizualizacja terenu
            self.frame.clear()

            # Dodaj wygładzoną powierzchnię ze zaktualizowaną teksturą
            mesh_actor = self.frame.add_mesh(
                extruded,
                scalars='Altitude',
                cmap='viridis',
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

            self.frame.add_mesh(boundary_edges, color='black', line_width=2)
            self.frame.add_mesh(vertical_edges, color='black', line_width=2)
            
            # Ustawienia oświetlenia dla lepszej wizualizacji terenu
            # Główne światło sceny
            main_light = pv.Light(light_type='scenelight')
            main_light.intensity = 0.5
            main_light.position = [0, 0, 1]
            main_light.focal_point = [0, 0, 0]
            self.frame.add_light(main_light)

            main_light = pv.Light(light_type='headlight')
            main_light.intensity = 0.3
            self.frame.add_light(main_light)
            
            main_light = pv.Light(light_type='cameralight')
            main_light.intensity = 0.85
            self.frame.add_light(main_light)
            
            # Siatka pomocnicza
            self.frame.show_grid(
                grid=True,
                location='outer',
                all_edges=True,
                xtitle='X [m]',
                ytitle='Y [m]',
                ztitle='Z [m]',
                font_size=12,
                padding=0.2,
                fmt='%d',
            )
            
            # Referencje do oryginalnych kolormap
            self.altitude_cmap = 'terrain'
            self.slope_cmap = 'fire'
            
            # Zmienna do śledzenia aktualnego skalara
            self.current_scalar = 'Altitude'

            # Generowanie iso-linii na wygładzonym modelu
            n_contours = 15
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
            
            ## Zapisz znormalizowaną chmurę punktów
            self.normalized_cloud = point_cloud

            # Dodaj przycisk przełączania punktów
            self.frame.add_checkbox_button_widget(
                self.toggle_points,  # Używamy metody klasy
                value=False,
                size=30,
                position=(10, 25),
                color_on='green',
                color_off='red',
                background_color='white'
            )

            # Dodaj tekst opisujący przycisk
            self.frame.add_text(
                'Pokaż punkty pomiarowe',
                position=(50, 25),  # Dopasuj pozycję do przycisku
                font_size=9,
                color='black',
                shadow=True
            )
            
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

    def analyze_slope(self, smoothed, point):
        """
        Analizuje nachylenie w konkretnym punkcie modelu
        
        Args:
            smoothed: Model terenu (PyVista DataSet)
            point: Punkt [x, y] dla którego chcemy znaleźć nachylenie
        """
        # Znajdź najbliższy punkt w modelu
        closest_point = smoothed.find_closest_point(point)
        cell_id = smoothed.find_closest_cell(point)
        
        # Pobierz nachylenie w tym punkcie
        terrain_slope = smoothed.compute_derivative(scalars='Altitude')
        gradient = terrain_slope['gradient'][cell_id]
        
        # Oblicz nachylenie w procentach
        slope_percent = np.sqrt(gradient[0]**2 + gradient[1]**2) * 100
        
        # Oblicz kierunek nachylenia (azymut)
        slope_direction = np.degrees(np.arctan2(gradient[1], gradient[0]))
        if slope_direction < 0:
            slope_direction += 360
            
        return {
            'point': point,
            'slope_percent': slope_percent,
            'slope_direction': slope_direction,
            'elevation': smoothed.points[closest_point][2]
        }

    def analyze_slope_area(self, smoothed, min_x, max_x, min_y, max_y):
        """
        Analizuje nachylenie w wybranym obszarze
        """
        # Znajdź wszystkie punkty w zadanym obszarze
        mask = ((smoothed.points[:, 0] >= min_x) & 
                (smoothed.points[:, 0] <= max_x) & 
                (smoothed.points[:, 1] >= min_y) & 
                (smoothed.points[:, 1] <= max_y))
        
        points_in_area = smoothed.points[mask]
        
        terrain_slope = smoothed.compute_derivative(scalars='Altitude')
        gradients = terrain_slope['gradient'][mask]
        
        slopes = np.sqrt(gradients[:, 0]**2 + gradients[:, 1]**2) * 100
        
        return {
            'mean_slope': np.mean(slopes),
            'max_slope': np.max(slopes),
            'min_slope': np.min(slopes),
            'std_slope': np.std(slopes),
            'points_analyzed': len(points_in_area)
        }

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
        processed_points = self.process_points(points, radius=1.5)
        
        # Stworzenie wygładzonego modelu powierzchni
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
        processed_points = self.process_points(points, radius=1.5)
        
        # Create visualization
        self.create_visualization(processed_points)

    def update_visualization_buttons(self):
        """Enable/disable buttons based on selection."""
        selected_items = self.session_list.selectedItems()
        enable_buttons = len(selected_items) > 0
        
        # Ustaw kolor tła dla wybranych elementów
        for i in range(self.session_list.count()):
            item = self.session_list.item(i)
            if item in selected_items:
                item.setBackground(QColor(255, 192, 255))  # Jasnoróżowe tło dla zaznaczonych
            else:
                item.setBackground(QColor(48, 48, 48))  # Przywróć domyślny kolor
        
        self.visualize_button_raw.setEnabled(enable_buttons)
        self.visualize_button.setEnabled(enable_buttons)
        self.visualize_button_interpolation.setEnabled(enable_buttons)
        #self.visualize_button_kriging.setEnabled(enable_buttons)
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
