# GNN4IoT
### This is the repository for the collection of applying Graph Neural Networks in Internet of Things (IoT).

#### If you find this repository helpful, you may consider cite our work:
* Guimin Dong, Mingyue Tang, Zhiyuan Wang, Jiechao Gao, Sikun Guo, Lihua Cai, Robert Gutierrez, Bradford Campbell, Laura E. Barnes, Mehdi Boukhechba, <b>Graph Neural Networks in IoT: A Survey</b>.

### We categorize GNNs in IoT into the following groups based on their semantics of graph modeling: Multi-agent Interaction, Human State Dynamics, and IoT Sensor Interconnection.

### Multi-agent Interaction:
* Casas, Sergio, et al. "Spagnn: Spatially-aware graph neural networks for relational behavior forecasting from sensor data." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020. [Link](https://ieeexplore.ieee.org/abstract/document/9196697?casa_token=Tit2yCbnIwgAAAAA:o6YT-OmwgvOFZcr2M1Vl1KCFq7vHpeQG7b8hFwvawVWacN-7-RKm2Q4Jl_0iGt0VLzRUVVWbKuE)


### Human State Dynamics:
* Dong, Guimin, et al. "Semi-supervised Graph Instance Transformer for Mental Health Inference." 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 2021. [Link](https://ieeexplore.ieee.org/abstract/document/9679981?casa_token=NZktpSySDKUAAAAA:9yAScp6BkRfYUUOwXfUmMWRg2TG6bEQJ3xnVSRMk0R1g2_TMUr_toVJyA9RXcawOHayjBbRFyKg)

### IoT Sensor Interconnection:
* Zhang, Weishan, et al. "Modeling IoT equipment with graph neural networks." IEEE Access 7 (2019): 32754-32764. [Link](https://ieeexplore.ieee.org/abstract/document/8658112)


### Relavant Public Dataset for GNN in IoT:
### Public Datasets

Dataset used or potential helpful in GNN-related research.

#### Human Acitivity Recognition (HAR)

| Name                                                         | Feature                                                      | Link                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NTU RGB+D                                                    | RGB videos, depth map sequences, 3D skeletal data, and infrared (IR) videos | [Link](https://rose1.ntu.edu.sg/dataset/actionRecognition/)  |
| [MobiAct](https://www.scitepress.org/Papers/2016/57924/57924.pdf) | accelerometer, gyroscope, orientation                        | [Link](http://www.bmi.teicrete.gr/)                          |
| [WISDM](https://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf) | accelerometer                                                | [Link](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+) |
| [MHEALTH](https://orestibanos.com/paper_files/banos_iwaal_2014.pdf) | accelerometer, gyroscope, magnetic, ecg                      | [Link](http://archive.ics.uci.edu/ml/datasets/mhealth+dataset) |
| PAMAP2                                                       | IMU hand, IMU chest, IMU ankle, heart rate                   | [Link](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring) |
| HHAR                                                         | accelerometer, gyroscope                                     | [Link](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition) |
| [USC-HAD](https://sipi.usc.edu/had/mi_ubicomp_sagaware12.pdf) | IMU, accelerometer, gyroscope, magnetometer                  | [Link](https://sipi.usc.edu/had/)                            |

#### Fall Detection

| Name     | Feature                                                      | Link                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TST V2   | depth frames and skeleton joints collected using Microsoft Kinect v2 | [Link](https://ieee-dataport.org/documents/tst-fall-detection-dataset-v2) |
| FallFree | Kinect camera combines the RGB color, depth, skeleton, infrared, body index into one single camera | Contact Author                                               |

#### Sleep Quality

| Name                                         | Feature                | Link                                  |
| -------------------------------------------- | ---------------------- | ------------------------------------- |
| Montreal Archive of Sleep Studies (MASS)     | polysomnography (PSG)  | [Link](http://ceams-carsm.ca/mass/)   |
| [ISRUC-SLEEP](https://sleeptight.isr.uc.pt/) | polysomnographic (PSG) | [Link](https://sleeptight.isr.uc.pt/) |

#### Air Quality

| Name                           | Feature                                                      | Link                                                         |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| KnowAir                        | temperature, boundary_layer_height, k_index, humidity, surface_pressure, total_precipitation, component_of_wind | [Link](https://github.com/shawnwang-tech/PM2.5-GNN)          |
| Beijing, Tianjing              | Hourly scaled dataset of pollutants (𝑃𝑀2.5, 𝑃𝑀10, 𝑁𝑂2,𝑆𝑂2,𝑂3,𝐶𝑂) from 76 station | [Link](http://urban-computing.com/data/Data-1.zip)           |
| Beijing Multi-Site Air-Quality | PM2.5, PM10, SO2: SO2, NO2, CO, O3, temperature, pressure (hPa), dew point temperature (degree Celsius), precipitation (mm), wind direction, wind speed (m/s), name of the air-quality monitoring site | [Link](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data) |

#### Water System

| Name                      | Feature                                                      | Link                                              |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| USGS                      | River segments that vary in length from 48 to 23,120 meters  | [Link](https://waterdata.usgs.gov/nwis)           |
| Water Calibration Network | Containing 388 nodes, 429 pipes, one reservoir, and seven tanks | [Link](https://uknowledge.uky.edu/wdst_models/2/) |

#### Soil

| Name        | Feature                                            | Link                                                         |
| ----------- | -------------------------------------------------- | ------------------------------------------------------------ |
| Spain       | 20 soil moisture stations from North-Western Spain | [Link](https://disc.gsfc.nasa.gov/information/documents?title=Hydrology%20Documentation) |
| Alabama     | 8 soil moisture stations from Alabama              | [Link](https://www.wcc.nrcs.usda.gov/scan/)                  |
| Mississippi | 5 soil moisture stations from Mississippi          | [Link](https://www.wcc.nrcs.usda.gov/scan/)                  |

#### Transportation

| Name        | Feature                                                      | Link                                                         |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GNN4Traffic | Repository for the collection of Graph Neural Network for Traffic Forecasting | [Link](https://github.com/jwwthu/GNN4Traffic)                |
| TLC Trip    | Origin-Destination demand taxi dataset, trip record data     | [Link](Taxi Trajectories for all the 442 taxis running in the city of Porto, in Portugal) |
| Kaggle Taxi | Taxi Trajectories for all the 442 taxis running in the city of Porto, in Portugal | [Link](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i) |

#### Autonomous Vehicle

| Name                  | Feature                                                      | Link                                                         |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| US Highway 101        | Vehicle trajectory data                                      | [Link](https://www.fhwa.dot.gov/publications/research/operations/07030/) |
| Interstate 80 Freeway | Vehicle trajectory data                                      | [Link](https://www.fhwa.dot.gov/publications/research/operations/06137/) |
| Stanford Drone        | Pedestrians, but also bicyclists, skateboarders, cars, buses, and golf carts trajectory data | [Link](https://cvgl.stanford.edu/projects/uav_data/)         |

#### Energy Prediction

| Name         | Feature                                                      | Link                                      |
| ------------ | ------------------------------------------------------------ | ----------------------------------------- |
| Pecan street | Minute-interval appliance-level customer electricity use from nearly 1,000 houses and apartments | [Link](https://dataport.pecanstreet.org/) |
