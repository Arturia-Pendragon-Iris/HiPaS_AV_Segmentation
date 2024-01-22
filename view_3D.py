import os
import vtk
import sys
from medpy import io
import numpy as np
sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
# import Tool_Functions.Functions as Functions


# def visualize_stl(stl_path):
#     Functions.visualize_stl(stl_path)


def convert_mha_to_stl(mha_path, stl_path=None, visualize=False):
    # mha_file_path = "E:/vtk_stl/LI(3).mha"  # 这是mha文件的路径
    # stl_file_path = "E:/vtk_stl/Li(3).stl"  # 这是保存stl文件的路径
    mha_file_path = mha_path
    stl_file_path = stl_path
    if stl_file_path is None and visualize is False:
        return None

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(mha_file_path)
    reader.Update()

    extra = vtk.vtkMarchingCubes()
    extra.SetInputConnection(reader.GetOutputPort())
    extra.SetValue(0, 1)

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(extra.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.SetMapper.ScalarVisibilityOff()
    # actor.GetProperty().SetColor(120 / 255, 176 / 255, 210 / 255)
    actor.GetProperty().SetColor(180 / 255, 180 / 255, 180 / 255)
    # actor.GetProperty().SetColor(250 / 255, 127 / 255, 111 / 255)
    # actor.GetProperty().SetColor(180 / 255, 182 / 255, 184 / 255)
    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600, 600)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    ren.AddActor(actor)
    ren.SetBackground(1.0, 1.0, 1.0)

    # Enable user interface interactor
    # 显示三维模型，关闭后再保存stl文件
    if visualize:
        iren.Initialize()
        renWin.Render()
        iren.Start()
    if stl_file_path is None:
        return None

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(extra.GetOutputPort())
    triangle.PassVertsOff()
    triangle.PassLinesOff()

    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputConnection(triangle.GetOutputPort())

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangle.GetOutputPort())

    triangle2 = vtk.vtkTriangleFilter()
    triangle2.SetInputConnection(clean.GetOutputPort())
    triangle2.PassVertsOff()
    triangle2.PassLinesOff()

    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetInputConnection(triangle2.GetOutputPort())
    stlWriter.SetFileName(stl_file_path)
    stlWriter.SetFileTypeToBinary()
    stlWriter.Write()

    return None


def save_numpy_as_stl(np_array, save_dict, stl_name, visualize=False, spacing=(1, 1, 1)):
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)

    if stl_name[-4::] == '.stl' or stl_name[-4::] == '.mha':
        stl_name = stl_name[:-4]

    np_array = np.transpose(np_array, (1, 0, 2))
    np_array[np_array < 0.5] = 0
    np_array[np_array >= 0.5] = 1
    np_array = np_array.astype("uint8")
    header = io.Header(spacing=spacing)
    # print("mha file path:", save_dict + stl_name + '.mha')
    io.save(np_array, save_dict + stl_name + '.mha', hdr=header, use_compression=True)

    stl_path = save_dict + stl_name + ".stl"
    convert_mha_to_stl(save_dict + stl_name + '.mha', stl_path, visualize=visualize)


def visualize_numpy_as_stl(numpy_array, temp_path='/home/chuy/Downloads/temp.stl'):
    # temp_path: we need to save numpy_array as .stl, then load .stl to visualize
    # numpy_array should be binary, like 1 means inside tracheae, 0 means outside tracheae
    save_dict = temp_path[:-len(temp_path.split('/')[-1])]
    stl_name = temp_path.split('/')[-1]
    # print(numpy_array.shape)
    save_numpy_as_stl(numpy_array, save_dict, stl_name, visualize=True)


def visualize_enhanced_channels(array_with_enhanced_channel_dict, save_dict):
    arrays_name_list = os.listdir(array_with_enhanced_channel_dict)
    for array_name in arrays_name_list:
        array_with_enhanced_channel = np.load(os.path.join(array_with_enhanced_channel_dict, array_name))['array']
        high_recall_mask = array_with_enhanced_channel[:, :, :, 1]
        save_numpy_as_stl(high_recall_mask, os.path.join(save_dict, 'high_recall/'),
                          array_name[:-4] + '_high_recall.stl')

        high_precision_mask = array_with_enhanced_channel[:, :, :, 2]
        save_numpy_as_stl(high_precision_mask, os.path.join(save_dict, 'high_precision/'),
                          array_name[:-4] + '_high_precision.stl')


def visualize_two_numpy(numpy_array_1, numpy_array_2):
    temp_path_1 = '/home/chuy/Downloads/temp_1.stl'
    temp_path_2 = '/home/chuy/Downloads/temp_2.stl'
    # temp_path: we need to save numpy_array as .stl, then load .stl to visualize
    # numpy_array should be binary, like 1 means inside tracheae, 0 means outside tracheae
    save_dict_1 = temp_path_1[:-len(temp_path_1.split('/')[-1])]
    stl_name_1 = temp_path_1.split('/')[-1]
    save_numpy_as_stl(numpy_array_1, save_dict_1, stl_name_1, visualize=False)

    save_dict_2 = temp_path_2[:-len(temp_path_2.split('/')[-1])]
    stl_name_2 = temp_path_2.split('/')[-1]
    save_numpy_as_stl(numpy_array_2, save_dict_2, stl_name_2, visualize=False)

    stl_visualization_two_file(temp_path_1, temp_path_2)


def stl_visualization_two_file(file_1, file_2):
    reader_1 = vtk.vtkSTLReader()
    reader_1.SetFileName(file_1)

    mapper_1 = vtk.vtkPolyDataMapper()
    mapper_1.SetInputConnection(reader_1.GetOutputPort())

    actor_1 = vtk.vtkActor()
    actor_1.SetMapper(mapper_1)
    # actor_1.GetProperty().SetColor(77/255, 155/255, 259/255)
    actor_1.GetProperty().SetColor(130 / 255, 176 / 255, 210 / 255)
    # actor_1.GetProperty().SetColor(158/255, 158/255, 158/255)
    actor_1.RotateX(90)
    # actor_1.RotateY(90)
    # actor_1.RotateZ(-90)

    reader_2 = vtk.vtkSTLReader()
    reader_2.SetFileName(file_2)

    mapper_2 = vtk.vtkPolyDataMapper()
    mapper_2.SetInputConnection(reader_2.GetOutputPort())

    actor_2 = vtk.vtkActor()
    actor_2.SetMapper(mapper_2)
    actor_2.GetProperty().SetColor(241/255, 157/255, 151/255)
    # actor_2.GetProperty().SetColor(250/255, 127 / 255, 111 / 255)
    # actor_2.SetPosition(-190, -140, -40)
    actor_2.RotateX(90)
    # actor_2.RotateY(90)
    # actor_2.RotateZ(-90)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    ren.AddActor(actor_1)
    ren.AddActor(actor_2)
    ren.SetBackground(1.0, 1.0, 1.0)

    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


def visualize_three_numpy(numpy_array_1, numpy_array_2, numpy_array_3):
    temp_path_1 = "/home/chuy/Downloads/temp_1.stl"
    temp_path_2 = "/home/chuy/Downloads/temp_2.stl"
    temp_path_3 = "/home/chuy/Downloads/temp_3.stl"
    # temp_path: we need to save numpy_array as .stl, then load .stl to visualize
    # numpy_array should be binary, like 1 means inside tracheae, 0 means outside tracheae
    save_dict_1 = temp_path_1[:-len(temp_path_1.split('/')[-1])]
    stl_name_1 = temp_path_1.split('/')[-1]
    save_numpy_as_stl(numpy_array_1, save_dict_1, stl_name_1, visualize=False)

    save_dict_2 = temp_path_2[:-len(temp_path_2.split('/')[-1])]
    stl_name_2 = temp_path_2.split('/')[-1]
    save_numpy_as_stl(numpy_array_2, save_dict_2, stl_name_2, visualize=False)

    save_dict_3 = temp_path_3[:-len(temp_path_3.split('/')[-1])]
    stl_name_3 = temp_path_3.split('/')[-1]
    save_numpy_as_stl(numpy_array_3, save_dict_3, stl_name_3, visualize=False)

    stl_visualization_three_file(temp_path_1, temp_path_2, temp_path_3)


def stl_visualization_three_file(file_1, file_2, file_3):
    reader_1 = vtk.vtkSTLReader()
    reader_1.SetFileName(file_1)

    mapper_1 = vtk.vtkPolyDataMapper()
    mapper_1.SetInputConnection(reader_1.GetOutputPort())

    actor_1 = vtk.vtkActor()
    actor_1.SetMapper(mapper_1)
    # actor_1.GetProperty().SetColor(77/255, 155/255, 259/255)
    actor_1.GetProperty().SetColor(250 / 255, 127 / 255, 111 / 255)
    # actor_1.GetProperty().SetColor(246 / 255, 202 / 255, 229 / 255)
    # actor_1.GetProperty().SetColor(130 / 255, 176 / 255, 210 / 255)
    # actor_1.RotateX(90)
    # actor_1.RotateY(90)
    # actor_1.RotateZ(-90)

    reader_2 = vtk.vtkSTLReader()
    reader_2.SetFileName(file_2)

    mapper_2 = vtk.vtkPolyDataMapper()
    mapper_2.SetInputConnection(reader_2.GetOutputPort())

    actor_2 = vtk.vtkActor()
    actor_2.SetMapper(mapper_2)
    actor_2.GetProperty().SetColor(130 / 255, 176 / 255, 210 / 255)
    # actor_2.GetProperty().SetColor(207 / 255, 234 / 255, 241 / 255)
    # actor_2.GetProperty().SetColor(150 / 255, 234 / 255, 241 / 255)
    # actor_2.GetProperty().SetColor(246 / 255, 202 / 255, 229 / 255)
    # actor_2.SetPosition(-190, -140, -40)
    # actor_2.RotateX(90)
    # actor_2.RotateY(90)
    # actor_2.RotateZ(-90)

    reader_3 = vtk.vtkSTLReader()
    reader_3.SetFileName(file_3)

    mapper_3 = vtk.vtkPolyDataMapper()
    mapper_3.SetInputConnection(reader_3.GetOutputPort())

    actor_3 = vtk.vtkActor()
    actor_3.SetMapper(mapper_3)
    actor_3.GetProperty().SetColor(200/255, 200/255, 200/255)
    # actor_2.SetPosition(-190, -140, -40)
    # actor_2.RotateX(90)
    # actor_2.RotateY(90)
    # actor_3.RotateZ(-90)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    ren.AddActor(actor_1)
    ren.AddActor(actor_2)
    ren.AddActor(actor_3)
    ren.SetBackground(1.0, 1.0, 1.0)

    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


def visualize_stl(stl_path):
    import vtk
    filename = stl_path

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetColor(130 / 255, 176 / 255, 210 / 255)
    # actor.GetProperty().SetColor(238 / 255, 123 / 255, 108 / 255)
    actor.GetProperty().SetColor(126 / 255, 161 / 255, 187 / 255)
    # actor.GetProperty().SetColor(216 / 255, 123 / 255, 111 / 255)
    # actor.RotateX(90)
    actor.RotateY(-90)
    # actor.RotateZ(-90)
    # actor.GetProperty().SetColor(200 / 255, 203 / 255, 127 / 255)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    ren.AddActor(actor)
    ren.SetBackground(1.0, 1.0, 1.0)

    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()