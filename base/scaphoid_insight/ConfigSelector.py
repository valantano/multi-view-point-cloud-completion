import os

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtCore import pyqtSignal


class FolderSelectorWidget(QWidget):
    """
    A widget to select a configuration and experiment from a folder structure.
    It allows the user to navigate through different levels of folders and select a specific experiment.
    Emits a signal with the selected configuration class name, configuration name, and experiment name.
    Is used by Insighter

    UI Layout:
    +----------------+-----------------------------------------+-------------------+------------+
    | BaselineCfgs ^ | ScaphoidPointAttN_Baseline_Min_dorsal ^ | FBaseMinDorsal0 ^ | Load Model |
    +----------------+-----------------------------------------+-------------------+------------+
    Selected Path: /home/valantano/mt/repository/base/mvpcc_experiments/BaselineCfgs/ScaphoidPointAttN_Baseline_Min_dorsal/FBaseMinDorsal0
    """
    # Signal to emit the final selected full path
    expSelected = pyqtSignal(str, str, str)

    def __init__(self, parent=None, exp_path=None):
        super().__init__(parent)

        # Store the experiment path for folder scanning
        self.exp_path = exp_path
        # Dictionary to hold the hierarchical folder structure
        self.folder_structure = {}

        # Layout - horizontal layout for the widget
        layout = QHBoxLayout(self)

        # UI Components
        # Three combo boxes for hierarchical selection (Level 1 -> Level 2 -> Level 3)
        self.level1Combo = QComboBox()
        self.level2Combo = QComboBox()
        self.level3Combo = QComboBox()
        # Button to trigger model loading
        self.loadModelButton = QPushButton("Load Model")
        # Label to display the currently selected path
        self.resultLabel = QLabel("Selected Path:")

        # Add widgets to the horizontal layout
        layout.addWidget(self.level1Combo)
        layout.addWidget(self.level2Combo)
        layout.addWidget(self.level3Combo)
        layout.addWidget(self.loadModelButton)
        layout.addWidget(self.resultLabel)
        

        # Connect signals to their respective slots
        # When level 1 selection changes, populate level 2 options
        self.level1Combo.currentTextChanged.connect(self.populate_level2)
        # When level 2 selection changes, populate level 3 options
        self.level2Combo.currentTextChanged.connect(self.populate_level3)
        # When level 3 selection changes, update the displayed path
        self.level3Combo.currentTextChanged.connect(self.update_selected_path)
        # When load model button is clicked, emit the selection signal
        self.loadModelButton.clicked.connect(self.select_exp)

        # Initialize the widget by loading the folder structure
        self.load_folder_structure()

    def select_exp(self):
        """
        Emit signal with the currently selected experiment configuration.
        This method is called when the Load Model button is clicked.
        """
        config_class_name = self.level1Combo.currentText()
        config_name = self.level2Combo.currentText()
        exp_name = self.level3Combo.currentText()
        self.expSelected.emit(config_class_name, config_name, exp_name)


    def load_folder_structure(self):
        """
        Initialize the folder structure by scanning the experiment path.
        Populates the first level combo box and sets up initial selections.
        """
        self.folder_structure = self.scan_folders(self.exp_path)
        self.populate_level1()
        self.update_selected_path()
        self.select_exp()

    def scan_folders(self, exp_path):
        """
        Recursively scan the experiment path to build a 3-level hierarchical structure.
        
        Args:
            exp_path: Root path to scan for experiments
            
        Returns:
            dict: Nested dictionary representing the folder structure
                  {level1: {level2: [level3_folders]}}
        """
        structure = {}
        for dirpath, dirnames, _ in os.walk(exp_path):
            # Get relative path from the root experiment path
            rel_path = os.path.relpath(dirpath, exp_path)
            parts = rel_path.split(os.sep)

            # Level 1: Top-level directories (config classes)
            if len(parts) == 1 and parts[0] != ".":
                structure[parts[0]] = {}
            # Level 2: Second-level directories (config names)
            elif len(parts) == 2:
                level1, level2 = parts
                structure.setdefault(level1, {}).setdefault(level2, [])
            # Level 3: Third-level directories (experiment names)
            elif len(parts) == 3:
                level1, level2, level3 = parts
                structure.setdefault(level1, {}).setdefault(level2, []).append(level3)

        return structure

    def populate_level1(self):
        """
        Populate the first level combo box with top-level folder names.
        Clears all combo boxes and adds sorted level 1 options.
        """
        self.level1Combo.clear()
        self.level2Combo.clear()
        self.level3Combo.clear()
        self.level1Combo.addItems(sorted(self.folder_structure.keys()))
        self.update_selected_path()

    def populate_level2(self, level1):
        """
        Populate the second level combo box based on level 1 selection.
        
        Args:
            level1: Selected text from the first combo box
        """
        self.level2Combo.clear()
        self.level3Combo.clear()
        level2_dict = self.folder_structure.get(level1, {})
        self.level2Combo.addItems(sorted(level2_dict.keys()))
        self.update_selected_path()

    def populate_level3(self, level2):
        """
        Populate the third level combo box based on level 1 and level 2 selections.
        Filters out 'TFBoard' directories from the options.
        
        Args:
            level2: Selected text from the second combo box
        """
        level1 = self.level1Combo.currentText()
        level3_list = self.folder_structure.get(level1, {}).get(level2, [])
        self.level3Combo.clear()
        # Remove 'TFBoard' from the list if it exists (likely TensorBoard logs)
        level3_list = [l for l in level3_list if l != 'TFBoard']
        self.level3Combo.addItems(sorted(level3_list))
        self.update_selected_path()

    def update_selected_path(self):
        """
        Update the result label to show the complete selected path.
        Only displays the path if all three levels are selected.
        """
        level1 = self.level1Combo.currentText()
        level2 = self.level2Combo.currentText()
        level3 = self.level3Combo.currentText()
        if all([level1, level2, level3]):
            full_path = os.path.join(self.exp_path, level1, level2, level3)
            self.resultLabel.setText(f"Selected Path: {full_path}")
            
