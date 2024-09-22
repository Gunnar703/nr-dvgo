import os
import bpy
import json
import bmesh
import mathutils
import numpy as np


def initialize_beam_grid(mesh, length, width, height, num_subdivisions):
    vertex_xcoords = np.array([-1, 1]) * width / 2
    vertex_ycoords = np.array([-1, 1]) * height / 2
    vertex_zcoords = np.linspace(-1, 1, num_subdivisions) * length / 2

    vertex_xcoords, vertex_ycoords, vertex_zcoords = np.meshgrid(
        vertex_xcoords, vertex_ycoords, vertex_zcoords, indexing="ij"
    )

    all_vertices = []
    for k in range(num_subdivisions):
        # create all four vertices
        vertex_coords_ll = [
            vertex_xcoords[0, 0, k],
            vertex_ycoords[0, 0, k],
            vertex_zcoords[0, 0, k],
        ]
        vertex_coords_ul = [
            vertex_xcoords[0, 1, k],
            vertex_ycoords[0, 1, k],
            vertex_zcoords[0, 1, k],
        ]
        vertex_coords_ur = [
            vertex_xcoords[1, 1, k],
            vertex_ycoords[1, 1, k],
            vertex_zcoords[1, 1, k],
        ]
        vertex_coords_lr = [
            vertex_xcoords[1, 0, k],
            vertex_ycoords[1, 0, k],
            vertex_zcoords[1, 0, k],
        ]
        vertex_ll = mesh.verts.new(vertex_coords_ll)
        vertex_ul = mesh.verts.new(vertex_coords_ul)
        vertex_ur = mesh.verts.new(vertex_coords_ur)
        vertex_lr = mesh.verts.new(vertex_coords_lr)

        all_vertices.append([vertex_ll, vertex_ul, vertex_ur, vertex_lr])

        # create faces
        if k >= 1:
            # left face
            left_corners = [
                all_vertices[k - 1][0],
                all_vertices[k - 1][1],
                all_vertices[k][1],
                all_vertices[k][0],
            ]
            mesh.faces.new(left_corners)

            # right face
            right_corners = [
                all_vertices[k - 1][2],
                all_vertices[k - 1][3],
                all_vertices[k][3],
                all_vertices[k][2],
            ]
            mesh.faces.new(right_corners)

            # front face
            front_corners = [
                all_vertices[k - 1][0],
                all_vertices[k - 1][3],
                all_vertices[k][3],
                all_vertices[k][0],
            ]
            mesh.faces.new(front_corners)

            # rear face
            rear_corners = [
                all_vertices[k - 1][1],
                all_vertices[k - 1][2],
                all_vertices[k][2],
                all_vertices[k][1],
            ]
            mesh.faces.new(rear_corners)

    # top and bottom faces
    mesh.faces.new(all_vertices[0])  # bottom
    mesh.faces.new(all_vertices[-1])  # top
    return mesh

def attach_material_to_beam(beam_object):
    bpy.context.view_layer.objects.active = beam_object

    # new material slot
    bpy.ops.object.material_slot_add()

    # new material
    material = bpy.data.materials.new(name="CustomMaterial")

    # change material color to black
    material.diffuse_color = (0, 61 / 255, 118 / 255, 1.0)
    material.specular_intensity = 0

    # assign new material to material slot
    beam_object.active_material = material

def shapeFunctions(length):
    def N1(x):
        return (1 - 3 * (x / length) ** 2 + 2 * (x / length) ** 3) * (
            (0 <= x) & (x <= length)
        )

    def N2(x):
        return (x * (1 - x / length) ** 2) * ((0 <= x) & (x <= length))

    def N3(x):
        return (3 * (x / length) ** 2 - 2 * (x / length) ** 3) * (
            (0 <= x) & (x <= length)
        )

    def N4(x):
        return (x**2 / length * (x / length - 1)) * ((0 <= x) & (x <= length))

    return N1, N2, N3, N4

def interpolateNodalDisplacements(nodal_displacements, position, element_length):
    local_coordinate = position - np.floor(position / element_length) * element_length
    element_number = np.floor(position / element_length).astype(np.int8)

    if local_coordinate % element_length == 0 and position != 0:
        element_number -= 1
        local_coordinate = element_length

    N1, N2, N3, N4 = shapeFunctions(element_length)
    interpolated_displacement = (
        N1(local_coordinate) * nodal_displacements[2 * element_number]
        + N2(local_coordinate) * nodal_displacements[2 * element_number + 1]
        + N3(local_coordinate) * nodal_displacements[2 * element_number + 2]
        + N4(local_coordinate) * nodal_displacements[2 * element_number + 3]
    )

    return interpolated_displacement

def get_xyz_to_uvw(frame, nodal_displacements_all, element_length):
    nodal_displacements = nodal_displacements_all[:, frame]

    def xyz_to_uvw(x, y, z):
        v = interpolateNodalDisplacements(nodal_displacements, z + 0.5, element_length)
        return 0, v, 0

    return xyz_to_uvw

def get_original_vertices(mesh_data):
    n_vertices = len(mesh_data.vertices)
    all_vertices = np.zeros((n_vertices, 3))
    for i in range(n_vertices):
        x, y, z = mesh_data.vertices[i].co
        all_vertices[i, 0] = x
        all_vertices[i, 1] = y
        all_vertices[i, 2] = z
    return all_vertices

def deform_mesh(original_vertices, mesh_data, xyz_to_uvw):
    n_vertices = len(mesh_data.vertices)
    for i in range(n_vertices):
        x, y, z = original_vertices[i]
        u, v, w = xyz_to_uvw(x, y, z)
        mesh_data.vertices[i].co = (x + u, y + v, z + w)
    return mesh_data

def create_light(name, location, energy=1000):
    lamp_data = bpy.data.lights.new(name=name, type="POINT")
    lamp_data.energy = energy
    lamp_object = bpy.data.objects.new(name=name, object_data=lamp_data)
    bpy.context.collection.objects.link(lamp_object)
    lamp_object.location = location
    return lamp_data, lamp_object

def create_camera(name):
    camera_data = bpy.data.cameras.new(name)
    camera_object = bpy.data.objects.new(f"{name}_data", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    return camera_data, camera_object

def orient_camera(
    camera_object,
    position: mathutils.Vector,
    focal_point: mathutils.Vector,
    distance: float,
):
    direction = -(focal_point - position)
    rotation_quat = direction.to_track_quat("Z", "Y")

    camera_object.rotation_euler = rotation_quat.to_euler()
    camera_object.location = position
    return camera_object

def spherical_to_cartesian(radius, transverse, azimuth):
    x = radius * np.sin(azimuth) * np.cos(transverse)
    y = radius * np.sin(azimuth) * np.sin(transverse)
    z = radius * np.cos(azimuth)
    return x, y, z

def place_cameras(
    radius: float,
    focal_point: mathutils.Vector,
    divisions_transverse: int,
    divisions_azimuth: int,
):
    transverse = np.linspace(0, 2 * np.pi, divisions_transverse)
    azimuth = np.linspace(np.pi / 10, np.pi / 2, divisions_azimuth)
    transverse, azimuth = np.meshgrid(transverse, azimuth, indexing="ij")
    transverse = transverse.ravel()
    azimuth = azimuth.ravel()

    camera_objects = []
    camera_datas = []
    for i in range(azimuth.shape[0]):
        position = mathutils.Vector(
            spherical_to_cartesian(radius, transverse[i], azimuth[i])
        )
        camera_data, camera_object = create_camera(f"Camera_{i}")
        camera_object = orient_camera(camera_object, position, focal_point, radius)
        camera_objects.append(camera_object)
        camera_datas.append(camera_data)
    return camera_objects, camera_datas

def get_camera_intrinsics(scene, camera_object):
    camera_angle_x = camera_object.data.angle_x
    camera_angle_y = camera_object.data.angle_y

    # camera_object properties
    f_in_mm = camera_object.data.lens  # focal length in mm
    scale = scene.render.resolution_percentage / 100
    width_res_in_px = scene.render.resolution_x * scale  # width
    height_res_in_px = scene.render.resolution_y * scale  # height
    optical_center_x = width_res_in_px / 2
    optical_center_y = height_res_in_px / 2

    # pixel aspect ratios
    size_x = scene.render.pixel_aspect_x * width_res_in_px
    size_y = scene.render.pixel_aspect_y * height_res_in_px
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # sensor fit and sensor size (and camera_object angle swap in specific cases)
    if camera_object.data.sensor_fit == "AUTO":
        sensor_size_in_mm = (
            camera_object.data.sensor_height
            if width_res_in_px < height_res_in_px
            else camera_object.data.sensor_width
        )
        if width_res_in_px < height_res_in_px:
            sensor_fit = "VERTICAL"
            camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x
        elif width_res_in_px > height_res_in_px:
            sensor_fit = "HORIZONTAL"
        else:
            sensor_fit = "VERTICAL" if size_x <= size_y else "HORIZONTAL"

    else:
        sensor_fit = camera_object.data.sensor_fit
        if sensor_fit == "VERTICAL":
            sensor_size_in_mm = (
                camera_object.data.sensor_height
                if width_res_in_px <= height_res_in_px
                else camera_object.data.sensor_width
            )
            if width_res_in_px <= height_res_in_px:
                camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x

    # focal length for horizontal sensor fit
    if sensor_fit == "HORIZONTAL":
        sensor_size_in_mm = camera_object.data.sensor_width
        s_u = f_in_mm / sensor_size_in_mm * width_res_in_px
        s_v = f_in_mm / sensor_size_in_mm * width_res_in_px * pixel_aspect_ratio

    # focal length for vertical sensor fit
    if sensor_fit == "VERTICAL":
        s_u = f_in_mm / sensor_size_in_mm * width_res_in_px / pixel_aspect_ratio
        s_v = f_in_mm / sensor_size_in_mm * width_res_in_px

    camera_intr_dict = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": s_u,
        "fl_y": s_v,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": optical_center_x,
        "cy": optical_center_y,
        "w": width_res_in_px,
        "h": height_res_in_px,
        "aabb_scale": scene.aabb,
    }

    return camera_intr_dict

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def get_camera_extrinsics(camera_object, file_path):
    frame_data = {
        "file_path": f"{file_path}",
        "transform_matrix": listify_matrix(camera_object.matrix_world),
    }
    return frame_data

def init_json_dict(scene, camera_object):
    json_dict = {}
    intrinsics_dict = get_camera_intrinsics(scene, camera_object)
    json_dict.update(intrinsics_dict)
    json_dict["frames"] = []
    return json_dict

def append_capture_data(
    json_dict: dict, filepath_to_camera_data_dict: dict, frame: int = 0
):
    """
    filename_to_camera_data_dict: {
        "./path/to/file": camera_data
    }
    """
    for filepath, camera_data in filepath_to_camera_data_dict.items():
        extrinsics_dict = get_camera_extrinsics(camera_data, filepath)
        extrinsics_dict.update({"frame": frame})
        json_dict["frames"].append(extrinsics_dict)
    return json_dict

def save_json_dict(json_dict, save_file):
    with open(save_file, "w") as f:
        json.dump(json_dict, f, indent=4)

def render(camera_objects, output_filenames):
    filepath_to_camera_data_dict = {}

    # loop through all the cameras
    for camera_object, output_filename in zip(camera_objects, output_filenames):
        # set current camera as active
        bpy.context.scene.camera = camera_object

        # set the render path
        bpy.context.scene.render.filepath = output_filename

        # render
        bpy.ops.render.render(write_still=True)

        filepath_to_camera_data_dict[output_filename] = camera_object

    return filepath_to_camera_data_dict

# clear everything from the scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.outliner.orphans_purge()
bpy.ops.outliner.orphans_purge()
bpy.ops.outliner.orphans_purge()

# initialization
object_name = "beam"  # define the name of the beam object
mesh_data = bpy.data.meshes.new(f"{object_name}_data")  # create mesh data
mesh_object = bpy.data.objects.new(object_name, mesh_data)  # create mesh object
bpy.context.scene.collection.objects.link(
    mesh_object
)  # add the mesh object to the scene

# create cameras
focal_point = mathutils.Vector((0.0, 0.0, 0.0))
camera_objects, camera_datas = place_cameras(
    radius=2, focal_point=focal_point, divisions_transverse=10, divisions_azimuth=5
)

# set render resolution
bpy.context.scene.render.resolution_x = 256
bpy.context.scene.render.resolution_y = 256

# create lights
create_light("light1", (-2, -2, 3), energy=100)
create_light("light2", (-2, -2, -3), energy=100)
create_light("light3", (-2, 2, 3), energy=100)
create_light("light4", (-2, 2, -3), energy=100)
create_light("light5", (2, 2, 3), energy=100)
create_light("light6", (2, 2, -3), energy=100)
create_light("light7", (2, -2, 3), energy=100)
create_light("light8", (2, -2, -3), energy=100)

# make bg color black
bpy.context.scene.render.film_transparent = True

# attach texture to beam
attach_material_to_beam(mesh_object)

# initialize json dict for equilibrium scene
equilibrium_json_dict = init_json_dict(bpy.context.scene, camera_objects[0])

# create equilibrium scene
mesh = bmesh.new()
mesh = initialize_beam_grid(
    mesh, length=1.0, width=0.1, height=0.1, num_subdivisions=100
)
mesh.to_mesh(mesh_data)
mesh_data.update()

# render the equilibrium scene
data_folder = r"C:\Projects\dvgo\data"
equilibrium_scene_folder = os.path.join(data_folder, "beam_equilibrium")
os.makedirs(equilibrium_scene_folder, exist_ok=True)
equilibrium_scene_output_filenames = [
    f"Camera_{i}_equilibrium.png" for i in range(len(camera_objects))
]
equilibrium_scene_output_filenames = list(
    map(
        lambda x: os.path.join(equilibrium_scene_folder, "imgs", x),
        equilibrium_scene_output_filenames,
    )
)
filepath_to_camera_data_dict = render(
    camera_objects, equilibrium_scene_output_filenames
)

# write the equilibrium json
equilibrium_json_dict = append_capture_data(
    equilibrium_json_dict, filepath_to_camera_data_dict, frame=0
)
save_json_dict(
    equilibrium_json_dict,
    os.path.join(equilibrium_scene_folder, "transforms.json"),
)

# # render the animation
# nodal_displacements_all = np.loadtxt(
#     r"C:\Projects\tensorf\data\Nodal Displacements.txt"
# )
# animation_folder = os.path.join(data_folder, "beam_animation")
# os.makedirs(animation_folder, exist_ok=True)

# # initialize the animation json dict
# animation_json_dict = init_json_dict(bpy.context.scene, camera_objects[0])

# # get original beam vertex positions
# original_vertices = get_original_vertices(mesh_data)

# max_frames = 15
# num_frames_to_render = min(nodal_displacements_all.shape[1], max_frames)
# for frame in range(num_frames_to_render):
#     # deform mesh
#     xyz_to_uvw = get_xyz_to_uvw(frame, nodal_displacements_all, 0.5)
#     mesh_data = deform_mesh(original_vertices, mesh_data, xyz_to_uvw)
#     mesh_data.update()

#     # get filenames to output to
#     output_filenames = [
#         f"Camera_{i}_Frame_{frame}.png" for i in range(len(camera_objects))
#     ]
#     output_filenames = list(
#         map(lambda x: os.path.join(animation_folder, "imgs", x), output_filenames)
#     )
#     filepath_to_camera_data_dict = render(camera_objects, output_filenames)

#     # append to the animation json dict
#     animation_json_dict = append_capture_data(
#         animation_json_dict, filepath_to_camera_data_dict, frame=frame
#     )

# # write the animation json
# save_json_dict(
#     animation_json_dict, os.path.join(animation_folder, "transforms_train.json")
# )
# save_json_dict(
#     animation_json_dict, os.path.join(animation_folder, "transforms_test.json")
# )

# free memory
mesh.free()
