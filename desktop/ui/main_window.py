from PyQt6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Terrein Mapper!")
        self.setFixedSize(400, 200)

        layout = QVBoxLayout()

        button = QPushButton("Click me!")
        layout.addWidget(button)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
    
    def show(self):
        super().show()
        print("Hello, World!")
    
    def closeEvent(self, event):
        print("Goodbye, World!")
        event.accept()
    
    def keyPressEvent(self, event):
        print("Key pressed:", event.key())
    

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())