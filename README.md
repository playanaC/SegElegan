# SegElegan

# Image pipeline
![GitHub Logo](https://github.com/playanaC/SegElegan/blob/main/Data/imgs_paper/fig00.png)

# Proposed neural network model
![GitHub Logo](https://github.com/playanaC/SegElegan/blob/main/Data/imgs_paper/fig01.png)

# Post-processing method
![GitHub Logo](https://github.com/playanaC/SegElegan/blob/main/Data/imgs_paper/fig02.png)
(a) Worms gray-image (input data to the proposed neural network model). (b) Non-overlapping worms (individual worms and worms in contact between edges or ends). (c) Overlapping worms divided into parts (each color represents a part of each worm and overlapping part in light-green). (d) Prediction of individual worms, each color represents each worm (non-overlapping worms [1-16] and separate overlapping worms [17-26]).

# Worm prediction results
![GitHub Logo](https://github.com/playanaC/SegElegan/blob/main/Data/imgs_paper/fig07.png)
The images on the left show the data input to the proposed neuronal network model, the center images show the ground-truth labels (GT), and the images on the right show the result of the proposed method. The IoU results for whole image are 0.9122, 0.8215, 0.8806, while the results per worm are 1, 0.9063, 0.96875 (row1, 2, 3, respectively)


# References:
- Layana Castro Pablo E., Puchalt, J.C., Sánchez-Salmerón, A.-J. "Improving skeleton algorithm for helping *Caenorhabditis elegans* trackers". Scientific Reports (2020) [doi paper](https://doi.org/10.1038/s41598-020-79430-8).

- Layana Castro Pablo E., Puchalt, J.C., García Garví, A., Sánchez-Salmerón, A.-J. "*Caenorhabditis elegans* Multi-Tracker Based on a Modified Skeleton Algorithm". Sensors (2021) [doi paper](https://doi.org/10.3390/s21165622).

- Layana Castro Pablo E., García Garví, A., Navarro Moya, F., Sánchez-Salmerón, A.-J. "Skeletonizing *Caenorhabditis elegans* Based on U-Net Architectures Trained with a Multi-worm Low-Resolution Synthetic Dataset". International Journal of Computer Vision (2023) [doi paper](https://doi.org/10.1007/s11263-023-01818-6).

- Layana Castro Pablo E., García Garví, A., Sánchez-Salmerón, A.-J. "Automatic segmentation of *Caenorhabditis elegans skeletons* in worm aggregations using improved U-Net in low-resolution image sequences". Heliyon (2023) [doi paper](https://doi.org/10.1016/j.heliyon.2023.e14715).
