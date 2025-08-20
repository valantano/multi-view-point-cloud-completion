import numpy as np
import pyvista as pv
from contextlib import contextmanager


from PyQt5.QtWidgets import QWidget, QComboBox, QHBoxLayout, QLabel
from PyQt5 import QtWidgets

from base.scaphoid_insight.NetworkHandler import Network
from base.scaphoid_utils.AttnWeightsCollector import AttnWeightsCollector
from base.scaphoid_utils.logger import print_log


class AttnVisualizer(QWidget):

    def __init__(self, collector, plotter, logger):
        super().__init__()
        self.logger = logger

        self.collector: AttnWeightsCollector = collector
        self.plotter = plotter
        
        self.plotter.set_background('white')
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.enable_anti_aliasing()
        self.plotter.enable_point_picking(callback=self.on_point_clicked,
            use_mesh=True,
            show_message=True,
            show_point=True,
            color='red'
        )
        self.plotter.view_vector((1, 1, 1))  # Set initial view direction
        self.plotter.camera.up = (0, 0, 1)   # Lock up direction (e.g., Z-up)

        layout = QHBoxLayout(self)


        self.module_selector = QComboBox()
        self.layer_selector = QComboBox()
        self.query_id_selector = QtWidgets.QSpinBox()
        self.plot_btn = QtWidgets.QPushButton("Plot")

        layout.addWidget(QLabel("Module:")); layout.addWidget(self.module_selector)
        layout.addWidget(QLabel("Layer:")); layout.addWidget(self.layer_selector)
        layout.addWidget(QLabel("Query ID:")); layout.addWidget(self.query_id_selector)
        layout.addWidget(self.plot_btn)

        self.offset = np.array([0, 1, 0])

        self.query_id_selector.setRange(0, 0)
        self.query_id_selector.setValue(0)

        self.plot_btn.clicked.connect(self.__plot)
        self.module_selector.currentIndexChanged.connect(self.on_module_sel_change)
        self.layer_selector.currentIndexChanged.connect(self.on_layer_sel_change)
        self.query_id_selector.valueChanged.connect(self.on_query_id_change)

        self.module_id = 0
        self.layer_id = 0
        self.query_id = 0

        
        self.plotting_active = True
        self.emit_signals = True


    def reload(self):
        self.deactivate_plotting()      # needed because .clear will incoke currentTextChanged -> __plot will be called
        with self.supress_signals():
            self.plotter.clear()
            self.plotter.set_background('white')
            self.plotter.add_axes()
            self.plotter.show_grid()
            self.plotter.enable_anti_aliasing()

            self.plotter.view_vector((1, 1, 1))  # Set initial view direction
            self.plotter.camera.up = (0, 0, 1)   # Lock up direction (e.g., Z-up)

            self.query_id_selector.clear()
            self.layer_selector.clear()
            self.module_selector.clear()

            self.populate()
            
            print_log(self.collector.get_infos())
        self.activate_plotting()

        self.__plot()

    def re_plot(self):
        self.__plot()
        
    def activate_plotting(self):
        """
        """
        self.plotting_active = True

    def deactivate_plotting(self):
        """
        """
        self.plotting_active = False
    
    @contextmanager
    def supress_signals(self):
        """
        Context manager to suppress signals
        """
        try:
            self.emit_signals = False
            self.module_selector.blockSignals(True)
            self.layer_selector.blockSignals(True)
            self.query_id_selector.blockSignals(True)
            yield
        except Exception as e:
            print_log(f"Error occurred: {e}", self.logger, color='red')
            raise
        finally:
            self.emit_signals = True
            self.module_selector.blockSignals(False)
            self.layer_selector.blockSignals(False)
            self.query_id_selector.blockSignals(False)

    def populate(self):
        """
        Populate the module selector, layer selector and query id selector
        """
        self.populate_module_selector()
        self.populate_layer_selector()
        self.populate_query_id_selector()

    def populate_module_selector(self):
        with self.supress_signals():
            self.module_selector.clear()
            module_sel_items = [f"{v['name']}-{v['module_infos']}" for k, v in self.collector.get_infos().items()]
            self.module_selector.addItems(module_sel_items)
            self.module_id = self.module_selector.currentIndex()

    def populate_layer_selector(self):
        with self.supress_signals():
            self.layer_selector.clear()
            module = self.collector.get_infos()[self.module_id]
            if module is not None:
                layers = module['layers']    # {id: str} -> {0: 'Layer_0 -- weights: (1024, 4096) -- kv_inds: (4096,) - xyz_affil -- q_inds: (1024,) - xyz_affil', ...}
                layer_sel_items = [v for k, v in layers.items()]
                self.layer_selector.addItems(layer_sel_items)
                self.layer_id = self.layer_selector.currentIndex()
            
    def populate_query_id_selector(self):
        with self.supress_signals():
            self.query_id_selector.clear()
            module = self.collector.get_module(self.module_id)
            if module is not None:
                attn_indices = module.get_indices(self.layer_id)['q_ids']['inds']
                self.query_id_selector.setRange(0, len(attn_indices) - 1)
                self.query_id = self.ensure_query_id_in_range(self.query_id)
                self.query_id_selector.setValue(self.query_id)
            
    

    def on_module_sel_change(self):
        """
        If the module selector is changed, update the layer selector and query id selector
        """
        self.module_id = self.module_selector.currentIndex()
        self.populate_layer_selector()
        self.populate_query_id_selector()
        self.__plot()


    def on_layer_sel_change(self):
        """
        If the layer selector is changed, update the query id selector
        """
        self.layer_id = self.layer_selector.currentIndex()
        self.populate_query_id_selector()
        self.__plot()

    def on_query_id_change(self):
        """
        If the query id selector is changed, update the plot
        """
        self.query_id = self.ensure_query_id_in_range(self.query_id_selector.value())
        with self.supress_signals():
            self.query_id_selector.setValue(self.query_id)
        self.__plot()

    def ensure_query_id_in_range(self, query_idx):
        """
        Ensure that the query id is in range
        :param query_idx: query id
        :return: None
        """
        query_idx_new = 0
        module = self.collector.get_module(self.module_id)
        if module is not None:
            upper_bound = module.get_indices(self.layer_id)['q_ids']['inds'].shape[0]
            query_idx_new = min(query_idx, upper_bound - 1)
            query_idx_new = max(query_idx_new, 0)


        return query_idx_new


    def on_point_clicked(self, point, query_idx):

        self.query_id = self.ensure_query_id_in_range(query_idx)
        self.query_id_selector.setValue(self.query_id)

        if query_idx != self.query_id:
            print_log(f"Picked Point with ID: {query_idx} is out of range", self.logger, color='red')
        else:
            print_log(f"Picked Point with ID: {query_idx}", self.logger, color='blue')

        
    def get_module(self, module_id):
        """
        Get the module with the given id
        :param module_id: id of the module
        :return: module
        """
        module = None
        try: 
            module = self.collector.get_module(module_id)
        except IndexError as e:
            print_log(f"Module with id {module_id} not found", self.logger, color='red')
            
        return module


    def __plot(self):
        if not self.plotting_active:
            return

        module = self.get_module(self.module_id)
        if module is None:
            return
        
        in_out = self.collector.in_out

        attn_weights = module.get_weights(self.layer_id)
        attn_indices = module.get_indices(self.layer_id)



        k_v_input = in_out[attn_indices['k_v_ids']['pcd_ref']].squeeze().permute(1,0).cpu().numpy()
        if 'seeds' in attn_indices['q_ids']['pcd_ref']:
            q_input = in_out[attn_indices['q_ids']['pcd_ref']].squeeze().permute(1,0).cpu().numpy()
        else:
            q_input = in_out[attn_indices['q_ids']['pcd_ref']].squeeze().permute(1,0).cpu().numpy()

        print_log(f"Input: {k_v_input.shape}", self.logger, color='blue')
        print_log(f"Query Points: {q_input.shape}", self.logger, color='blue')
        print_log(f"Attention Weights: {attn_weights.shape}", self.logger, color='blue')

        
        k_v_points = k_v_input[attn_indices['k_v_ids']['inds']]
        k_v_cloud = pv.PolyData(k_v_points[:,:3] + self.offset)


        q_points = q_input[attn_indices['q_ids']['inds']]
        q_cloud = pv.PolyData(q_points[:,:3])

        k_v_cloud['attn'] = attn_weights[self.query_id]


        # self.plotter.clear()
        
        # remove previous added meshes
        self.plotter.remove_actor("attn")
        self.plotter.remove_actor("query_attn")
        self.plotter.remove_actor("query")
        self.plotter.remove_actor("query_points")

        # add new meshes
        self.plotter.add_mesh(q_cloud, name="query_points", color='blue', render_points_as_spheres=True, point_size=30, 
                              label="Query Points", opacity=0.8)
        
        self.plotter.add_mesh(k_v_cloud, name="attn", label="Attention Weights", scalars='attn', cmap='viridis', 
                              show_scalar_bar=True, render_points_as_spheres=True, point_size=30, opacity=1.0)

        highlight = pv.Sphere(radius=0.02, 
                              center=(q_input[attn_indices['q_ids']['inds'][self.query_id]][:3] + self.offset))
        self.plotter.add_mesh(highlight, color='red', name="query_attn", label="Query Point", opacity=0.4)

        highlight = pv.Sphere(radius=0.02, center=q_input[attn_indices['q_ids']['inds'][self.query_id]][:3])
        self.plotter.add_mesh(highlight, color='red', name="query", label="Query Point", opacity=0.4)

        self.plotter.add_legend()
        self.plotter.show()

