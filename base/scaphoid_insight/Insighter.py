
import os, sys
from enum import Enum
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow, BackgroundPlotter
from PyQt5.QtWidgets import QMenu, QToolBar, QLabel


# Add the main directory to sys.path such that submodules can be imported        
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, "..", '..'))
if main_dir not in sys.path:
    sys.path.append(main_dir)


from base.scaphoid_insight.InOutVisualizer import InOutVisualizer
from base.scaphoid_insight.AttnVisualizer import AttnVisualizer
from base.scaphoid_insight.ConfigSelector import FolderSelectorWidget
from base.scaphoid_insight.NetworkHandler import Network


EXPERIMENT_FOLDER = '/home/valantano/mt/repository/base/mvpcc_experiments'

class TabViewState(Enum):
    IN_OUT = 0
    ATTENTION = 1

        
class MyMainWindow(MainWindow):
    """
    Combines InOutVisualizer, ConfigSelector and AttnVisualizer in one QT Application.
    """

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)


        # create the frame
        self.frame = QtWidgets.QFrame()
        self.plotter = BackgroundPlotter()  # add the pyvista interactor object
        # self.plotter = QtInteractor(self.frame) # add the pyvista interactor object
        # self.plotter.interactor.RemoveObservers("KeyPressEvent")  # Disable keyboard interactions
        # self.plotter.interactor.RemoveObservers("MouseWheelEvent")  # Disable mouse wheel interactions


    
        self.signal_close.connect(self.plotter.close)

        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        meshMenu = mainMenu.addMenu('Mesh')

        
        ################################## Folder Selector ##################################
        self.folderSelector = FolderSelectorWidget(self, exp_path=EXPERIMENT_FOLDER)
        self.folderSelector.setFixedHeight(60)  # Make it toolbar-sized
        self.folderSelector.expSelected.connect(self.handle_exp_selected)
        
        toolbar = QToolBar("Folder Selector")
        toolbar.addWidget(self.folderSelector)
        self.addToolBar(toolbar)
        #####################################################################################


        self.network = Network()

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabsClosable(False)
        self.tab_widget.setMovable(True)
        self.tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        self.in_out_vis = InOutVisualizer(self.network, self.plotter, self.network.logger)
        self.attn_vis = AttnVisualizer(self.network.collector, self.plotter, self.network.logger)

        self.tab_widget.addTab(self.in_out_vis, "In/Out")
        self.tab_widget.addTab(self.attn_vis, "Attention")

        self.tab_widget.setCurrentIndex(TabViewState.IN_OUT.value)
        self.tab_widget.currentChanged.connect(self.switch_workaround)
        
        self.switch = False

        ####### Test sample switching buttons ########
        next_btn = QtWidgets.QPushButton("Next")
        current_btn = QtWidgets.QPushButton("Current")
        prev_btn = QtWidgets.QPushButton("Prev")
        self.sample_spinbox = QtWidgets.QSpinBox()
        self.sample_spinbox.setRange(0, len(self.network) - 1)  # Adjust range as needed


        next_btn.clicked.connect(self.next)
        current_btn.clicked.connect(self.current)
        prev_btn.clicked.connect(self.prev)
        self.sample_spinbox.valueChanged.connect(self.process_new_sample)


        h_layerout = QtWidgets.QHBoxLayout()
        h_layerout.addWidget(prev_btn)
        h_layerout.addWidget(current_btn)
        h_layerout.addWidget(next_btn)
        h_layerout.addWidget(QLabel("Sample ID:"))
        h_layerout.addWidget(self.sample_spinbox)


        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addWidget(self.plotter.interactor)
        self.frame.setLayout(vlayout)
        vlayout.addLayout(h_layerout)
        vlayout.addWidget(self.tab_widget)

        # self.folderSelector.select_exp()

        if show:
            self.showMaximized()

    def switch_workaround(self):
        print("Switching tabs")
        if self.tab_widget.currentIndex() == TabViewState.IN_OUT.value:
            self.in_out_vis.reload()
        elif self.tab_widget.currentIndex() == TabViewState.ATTENTION.value:
            self.attn_vis.reload()
        else:
            raise ValueError(f"Unknown tab index: {self.tab_widget.currentIndex()}")


    ########################### Event Handlers ###########################
    def handle_exp_selected(self, config_class_name, config_name, exp_name):
        self.network.reload(config_folder=config_class_name, config_name=config_name, exp_name=exp_name)
        self.process_new_sample()
        if self.tab_widget.currentIndex() == TabViewState.IN_OUT.value:
            self.in_out_vis.reload()
        elif self.tab_widget.currentIndex() == TabViewState.ATTENTION.value:
            self.attn_vis.reload()
        else:
            raise ValueError(f"Unknown tab index: {self.tab_widget.currentIndex()}")

    def process_new_sample(self, sample_id=None):
        if sample_id is None:
            sample_id = self.sample_spinbox.value()
        self.network.set_sample_id(sample_id)
        if self.tab_widget.currentIndex() == TabViewState.IN_OUT.value:
            self.in_out_vis.reload()
        elif self.tab_widget.currentIndex() == TabViewState.ATTENTION.value:
            self.attn_vis.reload()
        else:
            raise ValueError(f"Unknown tab index: {self.tab_widget.currentIndex()}")

    def next(self):
        new_sample_id =  self.network.get_valid_sample_id(self.network.get_sample_id()+1)
        self.sample_spinbox.setValue(new_sample_id)

    def current(self):
        new_sample_id = self.network.get_valid_sample_id(self.network.get_sample_id())
        self.sample_spinbox.setValue(new_sample_id)


    def prev(self):
        new_sample_id = self.network.get_valid_sample_id(self.network.get_sample_id()-1)
        self.sample_spinbox.setValue(new_sample_id)
    ######################################################################




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())