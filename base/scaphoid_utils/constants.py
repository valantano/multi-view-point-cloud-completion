import numpy as np


black_rgb = np.array([64, 64, 64])  # Slightly lighter black
blue_rgb = np.array([0, 153, 204])  # Softer, cooler blue
dark_blue_rgb = np.array([46, 46, 184])
yellow_rgb = np.array([179, 179, 0])  # darker yellow
red_rgb = np.array([255, 92, 51])  # Softer, cooler red
white_rgb = np.array([100, 100, 100])
green_rgb = np.array([0, 102, 0])

gt_unfocused_rgb = white_rgb
gt_rgb = black_rgb
partial_volar_rgb = green_rgb
partial_dorsal_rgb = dark_blue_rgb
pred_rgb = red_rgb
