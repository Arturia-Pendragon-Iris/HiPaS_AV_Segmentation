import os
import sys
import vtk
import numpy as np
from medpy import io

sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')

# Colors (R, G, B) in [0, 1]
COLOR_ARTERY = (241 / 255, 157 / 255, 151 / 255)
COLOR_VEIN = (130 / 255, 176 / 255, 210 / 255)
COLOR_LUNG = (200 / 255, 200 / 255, 200 / 255)
COLOR_DEFAULT = (126 / 255, 161 / 255, 187 / 255)


def _make_actor(color, rotate_x=0, rotate_y=0):
    actor = vtk.vtkActor()
    actor.GetProperty().SetColor(*color)
    if rotate_x:
        actor.RotateX(rotate_x)
    if rotate_y:
        actor.RotateY(rotate_y)
    return actor


def _render(*actors, window_size=(800, 800)):
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    for actor in actors:
        ren.AddActor(actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(*window_size)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    renWin.Render()
    iren.Start()


def _stl_reader_to_actor(filepath, color, rotate_x=0):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filepath)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = _make_actor(color, rotate_x=rotate_x)
    actor.SetMapper(mapper)
    return actor


def convert_mha_to_stl(mha_path, stl_path=None, visualize=False):
    if stl_path is None and not visualize:
        return

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(mha_path)
    reader.Update()

    mc = vtk.vtkMarchingCubes()
    mc.SetInputConnection(reader.GetOutputPort())
    mc.SetValue(0, 1)

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(mc.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(180 / 255, 180 / 255, 180 / 255)

    if visualize:
        _render(actor)

    if stl_path is not None:
        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputConnection(mc.GetOutputPort())
        triangle.PassVertsOff()
        triangle.PassLinesOff()

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(triangle.GetOutputPort())

        triangle2 = vtk.vtkTriangleFilter()
        triangle2.SetInputConnection(clean.GetOutputPort())
        triangle2.PassVertsOff()
        triangle2.PassLinesOff()

        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(triangle2.GetOutputPort())
        writer.SetFileName(stl_path)
        writer.SetFileTypeToBinary()
        writer.Write()


def save_numpy_as_stl(np_array, save_dir, stl_name, visualize=False, spacing=(1, 1, 1)):
    os.makedirs(save_dir, exist_ok=True)

    base = stl_name.rstrip('.stl').rstrip('.mha')
    mha_path = os.path.join(save_dir, base + '.mha')
    stl_path = os.path.join(save_dir, base + '.stl')

    arr = np.transpose(np_array, (1, 0, 2))
    arr = (arr >= 0.5).astype("uint8")
    io.save(arr, mha_path, hdr=io.Header(spacing=spacing), use_compression=True)
    convert_mha_to_stl(mha_path, stl_path, visualize=visualize)


def _numpy_to_temp_stl(np_array, temp_path):
    save_dir = os.path.dirname(temp_path)
    stl_name = os.path.basename(temp_path)
    save_numpy_as_stl(np_array, save_dir, stl_name, visualize=False)


def visualize_numpy_as_stl(numpy_array, temp_path='/home/chuy/Downloads/temp.stl'):
    _numpy_to_temp_stl(numpy_array, temp_path)
    visualize_stl(temp_path)


def visualize_two_numpy(numpy_array_1, numpy_array_2,
                        temp_dir='/home/chuy/Downloads'):
    path_1 = os.path.join(temp_dir, 'temp_1.stl')
    path_2 = os.path.join(temp_dir, 'temp_2.stl')
    _numpy_to_temp_stl(numpy_array_1, path_1)
    _numpy_to_temp_stl(numpy_array_2, path_2)
    stl_visualization_two_file(path_1, path_2)


def visualize_three_numpy(numpy_array_1, numpy_array_2, numpy_array_3,
                          temp_dir='/home/chuy/Downloads'):
    path_1 = os.path.join(temp_dir, 'temp_1.stl')
    path_2 = os.path.join(temp_dir, 'temp_2.stl')
    path_3 = os.path.join(temp_dir, 'temp_3.stl')
    _numpy_to_temp_stl(numpy_array_1, path_1)
    _numpy_to_temp_stl(numpy_array_2, path_2)
    _numpy_to_temp_stl(numpy_array_3, path_3)
    stl_visualization_three_file(path_1, path_2, path_3)


def stl_visualization_two_file(file_1, file_2):
    actor_1 = _stl_reader_to_actor(file_1, COLOR_VEIN, rotate_x=90)
    actor_2 = _stl_reader_to_actor(file_2, COLOR_ARTERY, rotate_x=90)
    _render(actor_1, actor_2)


def stl_visualization_three_file(file_1, file_2, file_3):
    actor_1 = _stl_reader_to_actor(file_1, COLOR_ARTERY)
    actor_2 = _stl_reader_to_actor(file_2, COLOR_VEIN)
    actor_3 = _stl_reader_to_actor(file_3, COLOR_LUNG)
    _render(actor_1, actor_2, actor_3)


def visualize_stl(stl_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())

    actor = _make_actor(COLOR_DEFAULT, rotate_y=-90)
    actor.SetMapper(mapper)
    _render(actor)
