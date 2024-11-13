from PyQt6.QtWidgets import QApplication

from .console import Console
from .core import Core
from .display import Display
from .main_window import MainWindow


class PCApp(QApplication):
    def __init__(self):
        super().__init__([])

        self.main_window = MainWindow()
        self.display = Display()
        self.console = Console()

        self.core = Core(self.console, self.display)

        self.main_window.attach(self.console)
        self.main_window.attach(self.display)
        self.main_window.show()