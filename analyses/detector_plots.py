import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def getModuleCoords(v_id, l_id, m_id):
    detectors = pd.read_csv('detectors.csv')
    coords = detectors[(detectors['volume_id'] == v_id) & (detectors['layer_id'] == l_id)
                       & (detectors['module_id'] == m_id)]

    c_vec = [coords.iloc[0]['cx'], coords.iloc[0]['cy'], coords.iloc[0]['cz']]
    hu = coords.iloc[0]['module_maxhu']
    hv = coords.iloc[0]['module_hv']

    def rotateCoords(vec):
        rotation_matrix = np.array([[coords.iloc[0]['rot_xu'],coords.iloc[0]['rot_xv'],coords.iloc[0]['rot_xw']],
                           [coords.iloc[0]['rot_yu'],coords.iloc[0]['rot_yv'],coords.iloc[0]['rot_yw']],
                           [coords.iloc[0]['rot_zu'],coords.iloc[0]['rot_zv'],coords.iloc[0]['rot_zw']]])
        return rotation_matrix.dot(vec)


    v1 = rotateCoords(np.array([-hu,-hv,0]))
    v2 = rotateCoords(np.array([hu,-hv,0]))
    v3 = rotateCoords(np.array([hu,hv,0]))
    v4 = rotateCoords(np.array([-hu,hv,0]))

    x = np.array([v1[0],v2[0],v3[0],v4[0]]) + c_vec[0]
    y = np.array([v1[1],v2[1],v3[1],v4[1]]) + c_vec[1]
    z = np.array([v1[2],v2[2],v3[2],v4[2]]) + c_vec[2]

    verts = [list(zip(x,y,z))]
    return verts

def plotWholeDetector():
    volume_ids = [7,8,9]
    detectors = pd.read_csv('detectors.csv')
    detectors['xyz'] = detectors[['cx', 'cy', 'cz']].values.tolist()

    volumes = detectors.groupby('volume_id')['xyz'].apply(list).to_frame()
    accept_volumes = detectors[detectors.volume_id.isin(volume_ids)]

    x_min, x_max = accept_volumes['cx'].min(), accept_volumes['cx'].max()
    y_min, y_max = accept_volumes['cy'].min(), accept_volumes['cy'].max()
    z_min, z_max = accept_volumes['cz'].min(), accept_volumes['cz'].max()

    volumes_layers = accept_volumes.groupby(['volume_id','layer_id'])['xyz'].apply(list).to_frame()
    fig = plt.figure(figsize=plt.figaspect(0.9))

    ax = plt.axes(projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')


    #pixel_detector = detectors[detectors.volume_id.isin([7,8,9])]
    pixel_detector = pixel_detector[pixel_detector.layer_id.isin([8])]
    for index, row in pixel_detector.iterrows():
        verts = getModuleCoords(row['volume_id'], row['layer_id'], row['module_id'])
        ax.add_collection3d(Poly3DCollection(verts, facecolors='silver', linewidths=1, edgecolor='black'), zs='z')

    plt.show()
