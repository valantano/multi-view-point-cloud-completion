import os, time


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import pyvista as pv
import imageio.v2 as imageio
import open3d as o3d
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox
from PyQt5 import QtWidgets

from submodules.PoinTr.extensions.emd import emd_module as emd
from PointAttn.utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D

from base.scaphoid_insight.NetworkHandler import Network
from base.scaphoid_utils.logger import print_log


class InOutVisualizer(QWidget):

    def __init__(self, network, plotter: QtInteractor, logger):
        super().__init__()
        self.logger = logger

        self.network:Network = network
        self.plotter = plotter
        
        self.plotter.set_background('white')

        self.plotter.enable_anti_aliasing()

       
        self.in_out_selector = QComboBox()
        self.in_out_selector.currentTextChanged.connect(self.re_plot_new_selection)

        self.but_create_gif = QPushButton("Create Gif")
        self.but_create_gif.clicked.connect(self.create_gif_rotation)

        self.but_create_stl = QPushButton("Create STL")
        self.but_create_stl.clicked.connect(self.create_stl)

        self.check_show_bounds = QtWidgets.QCheckBox("Show Bounds")
        self.check_show_bounds.setChecked(True)
        self.check_show_bounds.stateChanged.connect(self.handle_bounds)

        self.check_show_boundingbox = QtWidgets.QCheckBox("Show Bounding Box")
        self.check_show_boundingbox.setChecked(True)
        self.check_show_boundingbox.stateChanged.connect(self.handle_boundingbox)

        self.check_show_boundsAxes = QtWidgets.QCheckBox("Show Axes")
        self.check_show_boundsAxes.setChecked(False)
        self.check_show_boundsAxes.stateChanged.connect(self.handle_boundsAxes)

        self.check_show_errors = QtWidgets.QCheckBox("Show Errors")
        self.check_show_errors.setChecked(True)
        self.check_show_errors.stateChanged.connect(self.re_plot)

        self.cmap_selector = QComboBox()
        self.cmap_selector.addItems(['viridis', 'cividis', 'plasma', 'inferno', 'magma', 'binary', 'gray'])
        # self.cmap_selector.setItemText(2, "inferno")
        self.cmap_selector.currentTextChanged.connect(self.re_plot)

        self.cmap_coord_selector = QComboBox()
        self.cmap_coord_selector.addItems(['x', 'y', 'z'])
        self.cmap_coord_selector.setCurrentText('x')
        self.cmap_coord_selector.currentTextChanged.connect(self.re_plot)


        self.show_comparison_gt = QtWidgets.QCheckBox("Show GT")
        self.show_comparison_gt.setChecked(True)
        self.show_comparison_gt.stateChanged.connect(self.re_plot)

        self.reverse_transforms = QtWidgets.QCheckBox("Reverse Transforms")
        self.reverse_transforms.setChecked(True)
        self.reverse_transforms.stateChanged.connect(self.re_plot)

        self.comparison_gt_opacity = QSpinBox()
        self.comparison_gt_opacity.setRange(0, 100)
        self.comparison_gt_opacity.setValue(100)
        self.comparison_gt_opacity.setPrefix("GT Opacity 0-100: ")
        self.comparison_gt_opacity.valueChanged.connect(self.re_plot)

        self.spinbox_point_size = QSpinBox()
        self.spinbox_point_size.setRange(1, 60)
        self.spinbox_point_size.setValue(15)
        self.spinbox_point_size.setPrefix("Point Size: ")
        self.spinbox_point_size.valueChanged.connect(self.re_plot)

        self.label_min_error = QLabel("Min Error: ")
        self.label_max_error = QLabel("Max Error: ")
        

        layout = QHBoxLayout(self)
        layout.addWidget(QLabel("In/Out"))
        layout.addWidget(self.in_out_selector)
        layout.addWidget(self.but_create_gif)
        layout.addWidget(self.but_create_stl)
        layout.addWidget(self.check_show_bounds)
        layout.addWidget(self.check_show_boundingbox)
        layout.addWidget(self.check_show_boundsAxes)
        layout.addWidget(self.check_show_errors)
        layout.addWidget(QLabel("Colormap:"))
        layout.addWidget(self.cmap_selector)
        layout.addWidget(QLabel("CMAP Coord:"))
        layout.addWidget(self.cmap_coord_selector)
        layout.addWidget(self.show_comparison_gt)
        layout.addWidget(self.reverse_transforms)
        layout.addWidget(self.comparison_gt_opacity)
        layout.addWidget(self.spinbox_point_size)
        layout.addWidget(self.label_min_error)
        layout.addWidget(self.label_max_error)
        

        self.plotting_active = True

    def handle_bounds(self):
        if self.check_show_bounds.isChecked():
            self.check_show_boundsAxes.setChecked(False)
            # self.plotter.show_bounds()
            self.plotter.show_grid()
        else:
            self.plotter.remove_bounds_axes()

    def handle_boundingbox(self):
        if self.check_show_boundingbox.isChecked():
            self.plotter.add_bounding_box(line_width=2, color='black')
        else:
            self.plotter.remove_bounding_box(True)

    def handle_boundsAxes(self):
        if self.check_show_boundsAxes.isChecked():
            self.check_show_bounds.setChecked(False)
            self.plotter.add_axes()
            self.plotter.show_bounds()
            
        else:
            self.plotter.remove_bounds_axes()


    def reload(self):
        """
        """
        self.deactivate_plotting()      # needed because .clear will incoke currentTextChanged -> __plot will be called
        self.plotter.clear()
        
        self.handle_boundsAxes()
        self.handle_bounds()
        self.handle_boundingbox()

        tmp_in_out = self.in_out_selector.currentText()
        self.in_out_selector.clear()
        self.in_out_selector.addItems(self.network.get_in_out().keys())
        if tmp_in_out in self.network.get_in_out().keys():
            self.in_out_selector.setCurrentText(tmp_in_out)
        # else:
        #     continue
        #     self.in_out_selector.setCurrentText(self.network.get_in_out().keys()[0])
        self.activate_plotting()
        self.__plot()


    def re_plot_new_selection(self):
        if not self.plotting_active:
            return
        pcd_name = self.in_out_selector.currentText()
        self.deactivate_plotting()
        if pcd_name == 'gt':
            self.comparison_gt_opacity.setValue(100)
            self.spinbox_point_size.setValue(60)
            self.cmap_selector.setCurrentText('gray')
            self.cmap_coord_selector.setCurrentText('y')
        elif pcd_name == 'points-gt':
            self.comparison_gt_opacity.setValue(5)
            self.spinbox_point_size.setValue(40)
            self.cmap_selector.setCurrentText('viridis')
            self.cmap_coord_selector.setCurrentText('y')
        self.activate_plotting()
        self.__plot()

    def re_plot(self):
        self.__plot()

    def __normalize(self, np_arr):
        # Min-max normalization row-wise
        row_min = np_arr.min(axis=1, keepdims=True)
        row_max = np_arr.max(axis=1, keepdims=True)
        normalized_arr = (np_arr - row_min) / (row_max - row_min + 1e-8)
        return normalized_arr
    
    def activate_plotting(self):
        """
        """
        self.plotting_active = True

    def deactivate_plotting(self):
        """
        """
        self.plotting_active = False

    def create_gif_rotation(self):
        frame_dir = "./frames"
        os.makedirs(frame_dir, exist_ok=True)

        frame_paths = []

        self.plotter.open_gif("rotation.gif")
        for angle in range(0, 360, 5):
            self.plotter.camera.Azimuth(5)  # rotate camera by 5 degrees
            self.plotter.render()

            frame_path = os.path.join(frame_dir, f"frame_{angle:03d}.png")
            self.plotter.screenshot(frame_path)
            frame_paths.append(frame_path)
            
            QApplication.processEvents()
            time.sleep(0.05)  # controls rotation speed

        gif_folder_path = "./gifs"
        os.makedirs(gif_folder_path, exist_ok=True)
        # Find a unique filename if rotation.gif already exists
        base_gif_name = "rotation"
        gif_path = os.path.join(gif_folder_path, base_gif_name)

        gif_path = self.__get_unique_filename(gif_path, "gif")

        with imageio.get_writer(gif_path, mode='I', duration=0.03, loop=0) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))

    def __get_unique_filename(self, base_name, extension):
        """
        Generate a unique filename by appending a number if the file already exists.
        """
        count = 1
        unique_name = f"{base_name}.{extension}"
        while os.path.exists(unique_name):
            unique_name = f"{base_name}_{count}.{extension}"
            count += 1
        return unique_name

    def create_stl(self):
        pcd_name = self.in_out_selector.currentText()
        if '-' in pcd_name:
            print(f"Cannot create STL for multiple point clouds: {pcd_name}")
            return

        else:
            pcd_data = self.__get_pcd(pcd_name)
            points = pcd_data.squeeze().cpu().numpy()
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Estimate normals (required for Poisson)
            pcd.estimate_normals()
            
            best_mesh = None
            best_score = 0
            
            for alpha in [0.03, 0.05, 0.07, 0.1]:
                try:
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                    if len(mesh.triangles) > best_score:
                        best_mesh = mesh
                        best_score = len(mesh.triangles)
                except:
                    continue
            
            if best_mesh is None:
                print("Alpha shapes failed, trying Poisson reconstruction")
                best_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            # Use best_mesh instead of undefined mesh variable
            # mesh = best_mesh
            
            # Clean up the mesh
            mesh.compute_vertex_normals()
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Fill holes using different approaches
            mesh = self._fill_mesh_holes(mesh)
            
            # Smooth the mesh
            mesh = mesh.filter_smooth_simple(number_of_iterations=5)
            mesh.compute_vertex_normals()

            stl_path = os.path.join(os.getcwd(), f"{pcd_name}")
            stl_path = self.__get_unique_filename(stl_path, "stl")
            o3d.io.write_triangle_mesh(stl_path, mesh)
            print(f"STL file saved to {stl_path}")

    def _fill_mesh_holes(self, mesh):
        """Fill holes in the mesh using various techniques"""
        
        # Method 1: Try Poisson reconstruction with higher depth if alpha shapes were used
        if len(mesh.triangles) < 1000:  # If mesh seems incomplete
            print("Trying Poisson reconstruction for hole filling...")
            pcd_from_mesh = mesh.sample_points_uniformly(number_of_points=5000)
            pcd_from_mesh.estimate_normals()
            poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_from_mesh, depth=10, width=0, scale=1.1, linear_fit=False
            )
            if len(poisson_mesh.triangles) > len(mesh.triangles):
                mesh = poisson_mesh

        return mesh

    def __get_pcd(self, pcd_name):
        """
        Returns the point cloud data for the given pcd_name.
        """
        pcd = (self.network.get_in_out()[pcd_name][:, :3, :]).permute(0, 2, 1)
        return pcd


    def __plot(self):
        if not self.plotting_active:
            return
        print(self.network.get_in_out().keys())

        self.plotter.remove_actor("pcd")
        self.plotter.remove_actor("pcd1")
        self.plotter.remove_actor("pcd2")

        point_size = int(self.spinbox_point_size.value())
        pcd_name = self.in_out_selector.currentText()

        print(f"Plotting {pcd_name} with point size {point_size}")
        if '-' in pcd_name:
            pcd_1, pcd_2 = pcd_name.split('-')
            pcd1 = self.__get_pcd(pcd_1)
            pcd2 = self.__get_pcd(pcd_2)

            cham_loss = dist_chamfer_3D.chamfer_3DDist() 
            pred = pcd1
            gt = pcd2
            dist1, dist2, idx1, idx2 = cham_loss(gt, pred) # expects [b, n, 3] and [b, m, 3] tensors ### dist1: gt to output, dist2: output to gt, idx1: gt to output, idx2: output to gt
            # EMD = emd.emdModule()
            # dist, _ = EMD(pred, gt, 0.005, 10000)
            # emd_val = torch.mean(torch.sqrt(dist)) * 1000
            # print_log(f"EMD Value: {emd_val:.4f} mm", self.logger, color='blue')
            print_log(f"{pred.shape}, {gt.shape}", self.logger, color='blue')

            errors = dist2.sqrt().cpu() / self.network.scale  # scale errors to mm
            errors = errors.squeeze().numpy() 
            errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
            self.label_min_error.setText(f"Min Error: {errors.min():.4f} mm")
            self.label_max_error.setText(f"Max Error: {errors.max():.4f} mm")


            if self.reverse_transforms.isChecked():
                try:
                    pcd1 = self.network.reverse_transforms(pcd1, pcd_1)
                    pcd2 = self.network.reverse_transforms(pcd2, pcd_2)
                except Exception as e:
                    print(f"Error reversing transforms: {e}")
                    return

            pcd1 = pv.PolyData(pcd1.squeeze().cpu().numpy())
            pcd2 = pv.PolyData(pcd2.squeeze().cpu().numpy())
            pcd1['sum'] = np.sum(pcd1.points, axis=1)
            pcd2['sum'] = np.sum(pcd2.points, axis=1)
            pcd1['errors'] = errors


            custom_grays = plt.get_cmap("Greys")(np.linspace(0.2, 1.0, 256))[::-1]
            custom_binary = plt.get_cmap("binary")(np.linspace(0.0, 0.8, 256))[::-1]
            light_to_dark_gray = ListedColormap(custom_grays)
            light_to_dark_binary = ListedColormap(custom_binary)
            selected_cmap = self.cmap_selector.currentText()
            if selected_cmap == 'binary':
                selected_cmap = light_to_dark_binary
            elif selected_cmap == 'gray':
                selected_cmap = light_to_dark_gray

            scalars = 'errors' if self.check_show_errors.isChecked() else None
            show_scalar_bar = self.check_show_errors.isChecked()
            self.plotter.add_mesh(pcd1, name="pcd1", scalars=scalars, label=f"{pcd_1}", color='red', clim=[0.0, 2.0], 
                                  cmap=selected_cmap, show_scalar_bar=show_scalar_bar, render_points_as_spheres=True, 
                                  point_size=point_size, opacity=1.0)
            if self.show_comparison_gt.isChecked():
                self.plotter.add_mesh(pcd2, name="pcd2", label=f"{pcd_2}", color='red', show_scalar_bar=show_scalar_bar,
                                      render_points_as_spheres=True, point_size=point_size,
                                      opacity=self.comparison_gt_opacity.value() / 100.0)
            # self.plotter.add_legend()
            self.plotter.show()

        else:
            pcd, volar, dorsal = self.__get_pcd(pcd_name), self.__get_pcd('volar'), self.__get_pcd('dorsal')

            if self.reverse_transforms.isChecked():
                try:
                    pcd = self.network.reverse_transforms(pcd, pcd_name)
                    volar = self.network.reverse_transforms(volar, 'volar')
                    dorsal = self.network.reverse_transforms(dorsal, 'dorsal')
                except Exception as e:
                    print(f"Error reversing transforms: {e}")
                    return

            pcd = pv.PolyData(pcd.squeeze().cpu().numpy())
            volar_pcd = pv.PolyData(volar.squeeze().cpu().numpy())
            dorsal_pcd = pv.PolyData(dorsal.squeeze().cpu().numpy())

            if self.cmap_coord_selector.currentText() == 'x':
                coor_xyz = 0
            elif self.cmap_coord_selector.currentText() == 'y':
                coor_xyz = 1
            elif self.cmap_coord_selector.currentText() == 'z':
                coor_xyz = 2
            else:
                raise ValueError(f"Unknown coordinate for colormap: {self.cmap_coord_selector.currentText()}")
            
            pcd['sum'] = np.sum(pcd.points[:, coor_xyz].reshape(-1, 1), axis=1)
            volar_pcd['sum'] = np.sum(volar_pcd.points[:, coor_xyz].reshape(-1, 1), axis=1)
            dorsal_pcd['sum'] = np.sum(dorsal_pcd.points[:, coor_xyz].reshape(-1, 1), axis=1)


            custom_blues = plt.get_cmap("Blues")(np.linspace(0.5, 1.0, 256))[::-1]
            custom_greens = plt.get_cmap("Greens")(np.linspace(0.5, 1.0, 256))[::-1]
            custom_grays = plt.get_cmap("gray")(np.linspace(0.0, 0.8, 256))[::-1]
            custom_binary = plt.get_cmap("binary")(np.linspace(0.2, 1.0, 256))[::-1]

            light_to_dark_blue = ListedColormap(custom_blues)
            light_to_dark_green = ListedColormap(custom_greens)
            light_to_dark_gray = ListedColormap(custom_grays)
            light_to_dark_binary = ListedColormap(custom_binary)

            
            selected_cmap = self.cmap_selector.currentText()
            if selected_cmap == 'binary':
                selected_cmap = light_to_dark_binary
            elif selected_cmap == 'gray':
                selected_cmap = light_to_dark_gray


            self.plotter.add_mesh(pcd, name="pcd", label=f"{pcd_name}", scalars='sum', cmap=selected_cmap, 
                                  show_scalar_bar=False, render_points_as_spheres=True, point_size=point_size, 
                                  opacity=self.comparison_gt_opacity.value() / 100.0)
            if pcd_name == 'concat':
                self.plotter.add_mesh(volar_pcd, name="pcd1", label="Volar", cmap=light_to_dark_blue, scalars='sum', 
                                      render_points_as_spheres=True, point_size=point_size//1, opacity=1.0)
                self.plotter.add_mesh(dorsal_pcd, name="pcd2", label="Dorsal", cmap=light_to_dark_green, scalars='sum', 
                                      render_points_as_spheres=True, point_size=point_size//1, opacity=1.0)

            # self.plotter.add_legend()
            self.plotter.show()
            

