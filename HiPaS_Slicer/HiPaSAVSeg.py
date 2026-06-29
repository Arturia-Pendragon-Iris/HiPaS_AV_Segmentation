"""Minimal 3D Slicer module for HiPaS artery/vein segmentation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
RUNNER_PATH = MODULE_DIR / "hipas_inference.py"
DEFAULT_MODEL_DIR = REPO_ROOT
DEFAULT_SOURCE_DIR = REPO_ROOT / "Simple_AV_seg-main" / "Simple_AV_seg-main"


class HiPaSAVSeg(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "HiPaS AV Seg"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["HiPaS AV Seg contributors"]
        self.parent.helpText = "Minimal artery/vein segmentation demo for NIfTI CT volumes."
        self.parent.acknowledgementText = "Uses the unofficial HiPaS artery-vein segmentation implementation."


class HiPaSAVSegWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()

        self.logic = HiPaSAVSegLogic()

        parameters_collapsible_button = ctk.ctkCollapsibleButton()
        parameters_collapsible_button.text = "Parameters"
        self.layout.addWidget(parameters_collapsible_button)
        form_layout = qt.QFormLayout(parameters_collapsible_button)

        self.input_selector = slicer.qMRMLNodeComboBox()
        self.input_selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.input_selector.selectNodeUponCreation = True
        self.input_selector.addEnabled = False
        self.input_selector.removeEnabled = False
        self.input_selector.noneEnabled = False
        self.input_selector.showHidden = False
        self.input_selector.showChildNodeTypes = False
        self.input_selector.setMRMLScene(slicer.mrmlScene)
        self.input_selector.setToolTip("Input CT volume.")
        form_layout.addRow("Input volume:", self.input_selector)

        self.python_path_edit = self._path_row(
            form_layout,
            "External Python:",
            os.environ.get("HIPAS_PYTHON", ""),
            "Select the Python executable from the environment that has torch, monai, and nibabel.",
            file_mode=True,
        )
        self.model_dir_edit = self._path_row(
            form_layout,
            "Model directory:",
            str(DEFAULT_MODEL_DIR),
            "Directory containing lung.pth, main_AV.pth, and AV_stage_1.pth.",
            file_mode=False,
        )
        self.source_dir_edit = self._path_row(
            form_layout,
            "Source directory:",
            str(DEFAULT_SOURCE_DIR),
            "Directory containing models.py and frangi_gpu.py.",
            file_mode=False,
        )
        self.output_dir_edit = self._path_row(
            form_layout,
            "Output directory:",
            str(Path(slicer.app.temporaryPath) / "HiPaSAVSeg"),
            "Directory where output NIfTI masks will be written.",
            file_mode=False,
        )

        self.load_outputs_checkbox = qt.QCheckBox()
        self.load_outputs_checkbox.checked = True
        self.load_outputs_checkbox.toolTip = "Load generated labelmaps into the current Slicer scene."
        form_layout.addRow("Load outputs:", self.load_outputs_checkbox)

        self.apply_button = qt.QPushButton("Run")
        self.apply_button.toolTip = "Run HiPaS segmentation using the configured external Python environment."
        self.apply_button.enabled = False
        form_layout.addRow(self.apply_button)

        self.log_text = qt.QPlainTextEdit()
        self.log_text.readOnly = True
        self.log_text.minimumHeight = 180
        form_layout.addRow("Log:", self.log_text)

        self.input_selector.connect("currentNodeChanged(vtkMRMLNode*)", self.on_select)
        self.apply_button.connect("clicked(bool)", self.on_apply_button)

        self.layout.addStretch(1)
        self.on_select()

    def _path_row(self, form_layout, label, initial_path, tooltip, file_mode):
        widget = qt.QWidget()
        row = qt.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        line_edit = qt.QLineEdit()
        line_edit.text = initial_path
        line_edit.toolTip = tooltip
        browse_button = qt.QPushButton("Browse")
        browse_button.toolTip = tooltip
        row.addWidget(line_edit)
        row.addWidget(browse_button)
        form_layout.addRow(label, widget)

        def browse():
            if file_mode:
                selected = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), "Select Python executable", line_edit.text)
            else:
                selected = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Select directory", line_edit.text)
            if selected:
                line_edit.text = selected

        browse_button.connect("clicked(bool)", browse)
        return line_edit

    def on_select(self, *args):
        self.apply_button.enabled = self.input_selector.currentNode() is not None

    def append_log(self, text):
        self.log_text.appendPlainText(text.rstrip())
        slicer.app.processEvents()

    def on_apply_button(self):
        self.apply_button.enabled = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            input_node = self.input_selector.currentNode()
            result = self.logic.run(
                input_node=input_node,
                python_path=Path(self.python_path_edit.text.strip()),
                model_dir=Path(self.model_dir_edit.text.strip()),
                source_dir=Path(self.source_dir_edit.text.strip()),
                output_dir=Path(self.output_dir_edit.text.strip()),
                load_outputs=self.load_outputs_checkbox.checked,
                log_callback=self.append_log,
            )
            self.append_log("Done.")
            if result:
                self.append_log(json.dumps(result, indent=2))
        except Exception as exc:
            self.append_log("ERROR: " + str(exc))
            self.append_log(traceback.format_exc())
            slicer.util.errorDisplay(str(exc), windowTitle="HiPaS AV Seg")
        finally:
            qt.QApplication.restoreOverrideCursor()
            self.apply_button.enabled = self.input_selector.currentNode() is not None


class HiPaSAVSegLogic(ScriptedLoadableModuleLogic):
    def run(self, input_node, python_path, model_dir, source_dir, output_dir, load_outputs=True, log_callback=None):
        if input_node is None:
            raise ValueError("Select an input volume.")
        if not python_path.is_file():
            raise FileNotFoundError(f"External Python was not found: {python_path}")
        if not RUNNER_PATH.exists():
            raise FileNotFoundError(f"Inference runner was not found: {RUNNER_PATH}")
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory was not found: {model_dir}")
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory was not found: {source_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "hipas_input.nii.gz"
        if log_callback:
            log_callback(f"Saving input volume: {input_path}")
        if not slicer.util.saveNode(input_node, str(input_path)):
            raise RuntimeError(f"Failed to save input volume to: {input_path}")

        command = [
            str(python_path),
            str(RUNNER_PATH),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--source-dir",
            str(source_dir),
        ]
        if log_callback:
            log_callback("Running: " + subprocess.list2cmdline(command))

        completed = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO_ROOT))
        if completed.stdout and log_callback:
            log_callback(completed.stdout)
        if completed.stderr and log_callback:
            log_callback(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"Inference failed with exit code {completed.returncode}.")

        metadata_path = output_dir / "hipas_outputs.json"
        result = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if load_outputs:
            self.load_labelmaps(output_dir, input_node.GetName())
        return result

    def load_labelmaps(self, output_dir, input_name):
        for suffix, filename in (
            ("artery", "hipas_artery.nii.gz"),
            ("vein", "hipas_vein.nii.gz"),
            ("lung", "hipas_lung.nii.gz"),
        ):
            path = output_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Expected output was not found: {path}")
            success, node = slicer.util.loadLabelVolume(
                str(path),
                {"name": f"{input_name}_{suffix}"},
                returnNode=True,
            )
            if not success:
                raise RuntimeError(f"Failed to load output labelmap: {path}")
            display_node = node.GetDisplayNode()
            if display_node:
                display_node.SetOpacity(0.55)


class HiPaSAVSegTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.test_HiPaSAVSeg_smoke()

    def test_HiPaSAVSeg_smoke(self):
        self.delayDisplay("HiPaS AV Seg module loaded.")
