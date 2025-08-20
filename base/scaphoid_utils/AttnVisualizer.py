# import numpy as np
# import pyvista as pv
# import plotly.graph_objects as go
# import time

# from base.scaphoid_utils.logger import print_log
# from base.scaphoid_utils.AttnWeightsCollector import ModuleCollector



# class AttnVisualizer:

#     def __init__(self, collector, logger):
#         self.logger = logger
        
#         self.module_id = 0
#         self.in_out = collector.in_out
#         self.attn_module: ModuleCollector = collector.get_module(self.module_id)

#         self.layer_id = 0
#         self.query_idx = 0
#         self.calc_pass_through = False
#         if self.calc_pass_through:
#             self.calc_passthrough_attn()

#         self.plotter = pv.Plotter()
#         self.plotter.set_background('white')
#         self.plotter.add_axes()
#         self.plotter.show_grid()
#         self.plotter.enable_anti_aliasing()
#         self.plotter.enable_point_picking(callback=self.callback,
#             use_mesh=True,
#             show_message=True,
#             show_point=True,
#             color='red'
#         )
#         self.plotter.view_vector((1, 1, 1))  # Set initial view direction
#         self.plotter.camera.up = (0, 0, 1)   # Lock up direction (e.g., Z-up)


#         self.text = self.plotter.add_text("", position='ur')
#         self.update_text()

#         self.plotter.add_key_event("k", self.change_query_idx)
#         self.plotter.add_key_event("l", self.change_layer_id)
#         # self.p.add_key_event("b", self.switch_approach)
#         # self.p.add_key_event("d", self.switch_selection_mode)

#     def callback(self, point, query_idx):
#         """
#         Callback function to be called when a point is picked.
#         """
#         attn_indices = self.attn_module.get_indices(self.layer_id)
#         if query_idx < attn_indices['q_ids']['inds'].shape[0]:
#             self.query_idx = query_idx
#             print_log(f"Picked Point with ID: {query_idx}", self.logger, color='blue')
#             self.update_text()
#             self.plot()
#         else:
#             print_log(f"Picked Point with ID: {query_idx} is out of range", self.logger, color='red')
#             return
    
#     def update_text(self):
#         """
#         Helper func to update the text in the top right corner of the plotter.
#         """
        
#         query_text = f"Press 'k' to change query index: {self.query_idx}"

#         attn_indices = self.attn_module.get_indices(self.layer_id)

#         self.l_type = "Self Attention" if attn_indices['k_v_ids']['inds'].shape == attn_indices['q_ids']['inds'].shape else "Cross Attention"
        
#         layer_text = f"Press 'L' to change layer id: {self.layer_id} ({self.l_type})"
#         # select_str = f"'d' to switch selection-mode: {self.selection_mode}"
#         # approach_str = f"'b' to switch approach: {self.approach}"
#         entire_str = f"{query_text}\n{layer_text}"
#         self.text.set_text(position='ur', text=entire_str)


#     def change_query_idx(self):
#         self.query_idx = (self.query_idx + 1) % len(self.attn_module.get_weights(self.layer_id))
#         print_log(f"Query Index: {self.query_idx}", self.logger, color='blue')
#         self.update_text()
#         self.plot()
        
#     def change_layer_id(self):
#         self.layer_id = (self.layer_id + 1) % len(self.attn_module)
#         self.query_idx = self.query_idx % len(self.attn_module.get_indices(self.layer_id)['q_ids']['inds'])

#         print_log(f"Layer ID: {self.layer_id}", self.logger, color='blue')
#         self.update_text()
#         self.plot()

#     def normalize(self, np_arr):

#         # Min-max normalization row-wise
#         row_min = np_arr.min(axis=1, keepdims=True)
#         row_max = np_arr.max(axis=1, keepdims=True)
#         normalized_arr = (np_arr - row_min) / (row_max - row_min + 1e-8)
#         return normalized_arr



#     # def calc_passthrough_attn(self):
#     #     self.attn_module['passthrough'] = []

#     #     for i in range(len(self.attn_module['weights'])):
           
#     #         attn_weights = self.attn_module['weights'][i]


#     #         # normalize the attention weights between 0 and 1
#     #         # attn_weights = attn_weights - attn_weights.min()
#     #         # attn_weights = attn_weights / attn_weights.max()

#     #         attn_weights = self.normalize(attn_weights)

#     #         if i == 0:
#     #             self.attn_module['passthrough'].append(attn_weights)
#     #         else:
#     #             last_pass = self.attn_module['passthrough'][i-1]
#     #             attn_weights = attn_weights @ last_pass
#     #             attn_weights = self.normalize(attn_weights)
#     #             self.attn_module['passthrough'].append(attn_weights)


#     def plot(self):
#         attn_weights = self.attn_module.get_weights(self.layer_id)
#         attn_indices = self.attn_module.get_indices(self.layer_id)
#         # print_log(f"Attention Weights: {attn_weights}", self.logger, color='blue')
#         print_log(f"Attention Indices: {attn_indices}", self.logger, color='blue')
#         passthrough = self.attn_module['passthrough'][self.layer_id] if self.calc_pass_through else None

#         # print_log(f"Input: {in_out['input']}", logger, color='blue')
#         # input = in_out['input_logging'].squeeze().permute(1,0).cpu().numpy()

#         k_v_input = self.in_out[attn_indices['k_v_ids']['pcd_ref']].squeeze().permute(1,0).cpu().numpy()
#         if 'seeds' in attn_indices['q_ids']['pcd_ref']:
#             q_input = self.in_out[attn_indices['q_ids']['pcd_ref']].squeeze().permute(1,0).cpu().numpy()
#         else:
#             q_input = self.in_out[attn_indices['q_ids']['pcd_ref']].squeeze().permute(1,0).cpu().numpy()

#         print_log(f"Input: {k_v_input.shape}", self.logger, color='blue')
#         print_log(f"Query Points: {q_input.shape}", self.logger, color='blue')

        
#         k_v_points = k_v_input[attn_indices['k_v_ids']['inds']] if passthrough is None else k_v_input
#         k_v_cloud = pv.PolyData(k_v_points + np.array([0, 0, 1]))


#         q_points = q_input[attn_indices['q_ids']['inds']]
#         q_cloud = pv.PolyData(q_points)

#         if passthrough is not None:
#             k_v_cloud['attn'] = passthrough[self.query_idx]
#         else:
#             k_v_cloud['attn'] = attn_weights[self.query_idx]


#         # self.plotter.clear()
        
#         # remove previous added meshes
#         self.plotter.remove_actor("attn")
#         self.plotter.remove_actor("query_attn")
#         self.plotter.remove_actor("query")
#         self.plotter.remove_actor("query_points")

#         # add new meshes
#         self.plotter.add_mesh(q_cloud, name="query_points", color='blue', render_points_as_spheres=True, point_size=30, label="Query Points", opacity=0.8)
        
#         self.plotter.add_mesh(k_v_cloud, name="attn", label="Attention Weights", scalars='attn', cmap='viridis', show_scalar_bar=True, render_points_as_spheres=True, point_size=30, opacity=1.0)
        
#         highlight = pv.Sphere(radius=0.02, center=(q_input[attn_indices['q_ids']['inds'][self.query_idx]] + np.array([0, 0, 1])))
#         self.plotter.add_mesh(highlight, color='red', name="query_attn", label="Query Point", opacity=0.4)

#         highlight = pv.Sphere(radius=0.02, center=q_input[attn_indices['q_ids']['inds'][self.query_idx]])
#         self.plotter.add_mesh(highlight, color='red', name="query", label="Query Point", opacity=0.4)

#         self.plotter.add_legend()
#         self.plotter.show()
        





# # def plot_attn(query_idx, collector, base_model, logger):
# #     attn_module = collector.get_module(0)

# #     layer_id = 1
# #     attn_weights = attn_module['weights'][layer_id]
# #     attn_indices = attn_module['indices'][layer_id]

# #     in_out = base_model.module.in_out
# #     # print_log(f"Input: {in_out['input']}", logger, color='blue')
# #     input = in_out['input_logging'].squeeze().permute(1,0).cpu().numpy()



# #     k_v_points = input[attn_indices['k_v_ids']]
# #     q_points = input[attn_indices['q_ids']]

    
# #     attn = attn_weights[query_idx]
# #     points = k_v_points
# #     print_log(f"Points: {points.shape}", logger, color='blue')
# #     colors = attn

# #     print_log(f"Colors: {colors.shape}", logger, color='blue')

# #     scatter_points = go.Scatter3d(
# #         x=points[:, 0],
# #         y=points[:, 1],
# #         z=points[:, 2],
# #         mode='markers',
# #         marker=dict(
# #             size=5,
# #             color=colors,
# #             colorscale='Viridis',
# #             opacity=0.8,
# #             colorbar=dict(
# #                 title="Attention Weight",
# #                 tickmode="auto"
# #             )
# #         ),
# #         name='Input Points'
# #     )

# #     # affiliation = 'volar' if points[query_idx, 3] == 0 else 'dorsal'
# #     highlight_point = go.Scatter3d(
# #         x=[points[query_idx, 0]],
# #         y=[points[query_idx, 1]],
# #         z=[points[query_idx, 2]],
# #         mode='markers',
# #         marker=dict(
# #             size=10,
# #             color='red',
# #             opacity=0.2,
# #             symbol='circle'
# #         ),
# #         # name=f'Query Point ({affiliation})',
# #         name=f'Query Point',
# #     )
    
# #     fig = go.Figure(data=[scatter_points, highlight_point])
# #     fig.update_layout(
# #         scene=dict(aspectmode='data'),
# #         legend=dict(
# #             title="Legend",
# #             x=0,
# #             y=1,
# #             bgcolor='rgba(255,255,255,0.7)',
# #             bordercolor='Black',
# #             borderwidth=1
# #         )
# #     )
# #     fig.show()  # Ensure fig.show() is properly indented    
