import csv
import logging
import os

import numpy as np
import qt
import vtk

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from segmentation_core.anatomy import (
    DEFAULT_HU_RANGE,
    DEFAULT_MIN_SIZE_VOXELS,
    DEFAULT_OPENING_RADIUS_VOX,
    SINUS_REGIONS,
)
from segmentation_core.region_growing import segment_region


#
# SinusSegmentation
#


class SinusSegmentation(ScriptedLoadableModule):
    """Module registration/metadata."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Sinus Segmentation"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["SegmentStatistics"]
        self.parent.contributors = ["niyaziacer (github.com/niyaziacer)"]
        self.parent.helpText = (
            "Paranazal sinüsleri (maksiller, frontal, sfenoid, etmoid; sağ/sol) "
            "her biri için ayrı ayrı yerleştirilen bir tohum noktasından başlayan "
            "sınırlı bölge büyütme (region growing) ve morfolojik işlemlerle "
            "segmente eder; her sinüs için hacim (cm3) ve yüzey alanı (mm2) hesaplar. "
            "Ayrıntı için: https://github.com/niyaziacer/SinusSegmentation"
        )
        self.parent.acknowledgementText = (
            "3D Slicer'ın ScriptedLoadableModule ve SegmentStatistics altyapısı kullanılarak geliştirilmiştir."
        )


#
# SinusSegmentationWidget
#


class SinusSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.ui = None
        self.regionWidgets = {}  # region.id -> dict(checkbox, seedButton, statusLabel, fiducialNode)
        self.lastResultsRows = []
        self.placementObserverTag = None
        self.activePlacementRegionId = None

    def resourcePath(self, filename):
        return os.path.join(os.path.dirname(__file__), "Resources", filename)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath(os.path.join("UI", "SinusSegmentation.ui")))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = SinusSegmentationLogic()

        self._buildRegionRows()
        self._buildResultsTable()

        self.ui.inputVolumeSelector.currentNodeChanged.connect(self._updateApplyButtonState)
        self.ui.applyButton.clicked.connect(self.onApplyButton)
        self.ui.exportCsvButton.clicked.connect(self.onExportCsvButton)

        self._updateApplyButtonState()

    def cleanup(self):
        self._stopSeedPlacement()

    # -- dynamic UI construction -------------------------------------------------

    def _buildRegionRows(self):
        layout = qt.QVBoxLayout(self.ui.regionsContainer)
        layout.setContentsMargins(0, 0, 0, 0)

        for region in SINUS_REGIONS:
            rowWidget = qt.QWidget()
            rowLayout = qt.QHBoxLayout(rowWidget)
            rowLayout.setContentsMargins(0, 0, 0, 0)

            checkbox = qt.QCheckBox(region.name_tr)
            checkbox.setChecked(region.enabled_by_default)
            checkbox.setMinimumWidth(160)

            seedButton = qt.QPushButton("Tohum yerleştir")
            seedButton.setCheckable(True)

            statusLabel = qt.QLabel("tohum yok")
            statusLabel.setMinimumWidth(90)

            rowLayout.addWidget(checkbox)
            rowLayout.addWidget(seedButton)
            rowLayout.addWidget(statusLabel)
            layout.addWidget(rowWidget)

            seedButton.clicked.connect(lambda checked, rid=region.id: self.onPlaceSeedClicked(rid, checked))

            self.regionWidgets[region.id] = {
                "region": region,
                "checkbox": checkbox,
                "seedButton": seedButton,
                "statusLabel": statusLabel,
                "fiducialNode": None,
            }

    def _buildResultsTable(self):
        layout = qt.QVBoxLayout(self.ui.resultsContainer)
        layout.setContentsMargins(0, 0, 0, 0)

        table = qt.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Sinüs", "Hacim (cm3)", "Yüzey alanı (mm2)"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        layout.addWidget(table)

        self.resultsTable = table

    # -- seed placement ------------------------------------------------------

    def onPlaceSeedClicked(self, regionId, checked):
        if not checked:
            self._stopSeedPlacement()
            return

        # Only one region can be in "placing" mode at a time.
        for rid, widgets in self.regionWidgets.items():
            if rid != regionId:
                widgets["seedButton"].setChecked(False)

        self._stopSeedPlacement()

        widgets = self.regionWidgets[regionId]
        fiducialNode = widgets["fiducialNode"]
        if fiducialNode is None:
            fiducialNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", f"seed_{regionId}"
            )
            fiducialNode.SetMaximumNumberOfControlPoints(1)
            fiducialNode.CreateDefaultDisplayNodes()
            fiducialNode.GetDisplayNode().SetSelectedColor(*widgets["region"].color_rgb)
            widgets["fiducialNode"] = fiducialNode
        else:
            fiducialNode.RemoveAllControlPoints()

        self.activePlacementRegionId = regionId
        self.placementObserverTag = fiducialNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self._onSeedPlaced
        )

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetActivePlaceNodeID(fiducialNode.GetID())
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        widgets["statusLabel"].setText("tıklayın...")

    def _onSeedPlaced(self, caller, event):
        regionId = self.activePlacementRegionId
        if regionId is None:
            return
        widgets = self.regionWidgets[regionId]
        widgets["statusLabel"].setText("tohum OK")
        widgets["seedButton"].setChecked(False)
        self._stopSeedPlacement()

    def _stopSeedPlacement(self):
        if self.activePlacementRegionId is not None:
            widgets = self.regionWidgets.get(self.activePlacementRegionId)
            if widgets is not None and widgets["fiducialNode"] is not None and self.placementObserverTag is not None:
                widgets["fiducialNode"].RemoveObserver(self.placementObserverTag)
        self.placementObserverTag = None
        self.activePlacementRegionId = None

    # -- run / export ---------------------------------------------------------

    def _updateApplyButtonState(self):
        self.ui.applyButton.enabled = self.ui.inputVolumeSelector.currentNode() is not None

    def onApplyButton(self):
        volumeNode = self.ui.inputVolumeSelector.currentNode()
        if volumeNode is None:
            slicer.util.errorDisplay("Lütfen bir BT hacmi seçin.")
            return

        segmentationNode = self.ui.segmentationSelector.currentNode()
        if segmentationNode is None:
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentationNode.CreateDefaultDisplayNodes()
            segmentationNode.SetName("SinusSegmentation")
            self.ui.segmentationSelector.setCurrentNode(segmentationNode)
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

        huRange = (self.ui.huLowSpinBox.value, self.ui.huHighSpinBox.value)
        minSizeVoxels = int(self.ui.minSizeSpinBox.value)
        openingRadiusVox = int(self.ui.openingRadiusSpinBox.value)

        rowsToRun = [
            widgets for widgets in self.regionWidgets.values() if widgets["checkbox"].isChecked()
        ]
        self.ui.progressBar.maximum = max(1, len(rowsToRun))
        self.ui.progressBar.value = 0
        self.ui.statusLabel.setText("Çalışıyor...")
        slicer.app.processEvents()

        anyFailure = False
        for i, widgets in enumerate(rowsToRun):
            region = widgets["region"]
            fiducialNode = widgets["fiducialNode"]

            if fiducialNode is None or fiducialNode.GetNumberOfControlPoints() == 0:
                widgets["statusLabel"].setText("tohum yok - atlandı")
                anyFailure = True
                self.ui.progressBar.value = i + 1
                slicer.app.processEvents()
                continue

            result = self.logic.segmentOneRegion(
                volumeNode,
                segmentationNode,
                region,
                fiducialNode,
                huRange=huRange,
                minSizeVoxels=minSizeVoxels,
                openingRadiusVox=openingRadiusVox,
            )
            widgets["statusLabel"].setText(self.logic.describeResult(result))
            if not result.success:
                anyFailure = True

            self.ui.progressBar.value = i + 1
            slicer.app.processEvents()

        self.ui.statusLabel.setText(
            "Tamamlandı (bazı bölgeler başarısız oldu, ayrıntı için satır etiketlerine bakın)."
            if anyFailure
            else "Tamamlandı."
        )

        self._updateResultsTable(segmentationNode)

    def _updateResultsTable(self, segmentationNode):
        rows = self.logic.computeStatistics(segmentationNode)
        self.lastResultsRows = rows

        self.resultsTable.setRowCount(len(rows))
        for i, (name, volumeCm3, surfaceMm2) in enumerate(rows):
            self.resultsTable.setItem(i, 0, qt.QTableWidgetItem(name))
            volumeText = f"{volumeCm3:.3f}" if volumeCm3 is not None else "n/a"
            surfaceText = f"{surfaceMm2:.1f}" if surfaceMm2 is not None else "n/a"
            self.resultsTable.setItem(i, 1, qt.QTableWidgetItem(volumeText))
            self.resultsTable.setItem(i, 2, qt.QTableWidgetItem(surfaceText))

        self.ui.exportCsvButton.enabled = len(rows) > 0

    def onExportCsvButton(self):
        if not self.lastResultsRows:
            slicer.util.warningDisplay("Önce segmentasyonu çalıştırın.")
            return

        path = qt.QFileDialog.getSaveFileName(
            self.parent, "CSV olarak kaydet", "sinus_sonuclari.csv", "CSV files (*.csv)"
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sinus", "volume_cm3", "surface_mm2"])
            writer.writerows(self.lastResultsRows)

        slicer.util.infoDisplay(f"Kaydedildi: {path}")


#
# SinusSegmentationLogic
#


class SinusSegmentationLogic(ScriptedLoadableModuleLogic):
    """Slicer glue: converts between MRML nodes and the pure numpy
    segmentation_core algorithm, and computes per-segment statistics."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def ijkFromFiducial(self, volumeNode, fiducialNode):
        ras = [0.0, 0.0, 0.0]
        fiducialNode.GetNthControlPointPositionWorld(0, ras)

        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)
        ijkH = rasToIjk.MultiplyPoint((ras[0], ras[1], ras[2], 1.0))
        return int(round(ijkH[0])), int(round(ijkH[1])), int(round(ijkH[2]))

    def segmentOneRegion(self, volumeNode, segmentationNode, region, fiducialNode,
                          huRange=DEFAULT_HU_RANGE,
                          minSizeVoxels=DEFAULT_MIN_SIZE_VOXELS,
                          openingRadiusVox=DEFAULT_OPENING_RADIUS_VOX):
        volumeArray = slicer.util.arrayFromVolume(volumeNode)  # array axes are (k, j, i)
        spacingIJK = volumeNode.GetSpacing()  # (sx, sy, sz)
        spacingKJI = (spacingIJK[2], spacingIJK[1], spacingIJK[0])

        i, j, k = self.ijkFromFiducial(volumeNode, fiducialNode)
        seedKJI = (k, j, i)

        result = segment_region(
            volume_hu=volumeArray,
            spacing_mm=spacingKJI,
            seed_index=seedKJI,
            hu_range=huRange,
            crop_radius_mm=region.crop_radius_mm,
            min_size_voxels=minSizeVoxels,
            opening_radius_vox=openingRadiusVox,
        )

        if result.mask.any():
            segmentation = segmentationNode.GetSegmentation()
            segmentId = segmentation.GetSegmentIdBySegmentName(region.name_tr)
            if not segmentId:
                segmentId = segmentation.AddEmptySegment(region.id, region.name_tr, region.color_rgb)
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                result.mask.astype("uint8"), segmentationNode, segmentId, volumeNode
            )

        return result

    def describeResult(self, result):
        if result.success:
            return f"OK  {result.volume_cm3:.2f} cm3"
        reasons = {
            "seed_outside_bounds": "tohum hacim dışında",
            "seed_not_in_air": "tohum hava içinde değil",
            "too_small": f"çok küçük ({result.volume_voxels} voksel)",
            "possible_leak": f"olası sızıntı ({result.volume_cm3:.1f} cm3)",
        }
        return reasons.get(result.reason, result.reason or "başarısız")

    def computeStatistics(self, segmentationNode):
        """Returns a list of (segmentName, volume_cm3, surface_mm2) tuples,
        using Slicer's built-in SegmentStatistics module rather than a
        hand-rolled marching-cubes pass."""
        import SegmentStatistics

        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatistics.enabled", str(True))
        segStatLogic.getParameterNode().SetParameter("ClosedSurfaceSegmentStatistics.enabled", str(True))
        segStatLogic.computeStatistics()
        stats = segStatLogic.getStatistics()

        rows = []
        for segmentId in stats.get("SegmentIDs", []):
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            name = segment.GetName() if segment else segmentId
            volumeCm3 = stats.get((segmentId, "LabelmapSegmentStatistics.volume_cm3"))
            surfaceMm2 = stats.get((segmentId, "ClosedSurfaceSegmentStatistics.surface_mm2"))
            rows.append((name, volumeCm3, surfaceMm2))
        return rows


#
# SinusSegmentationTest
#


class SinusSegmentationTest(ScriptedLoadableModuleTest):
    """Standard Slicer self-test. Builds a synthetic phantom (no external
    data dependency) and checks the full node round-trip: volume -> seed
    fiducial -> segmentation -> computed volume close to the analytic value.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_SinusSegmentationPhantom()

    def test_SinusSegmentationPhantom(self):
        import math

        self.delayDisplay("Building synthetic phantom volume")

        shape = (60, 60, 60)  # (k, j, i)
        spacing = (0.5, 0.5, 0.5)
        center = (30, 30, 30)
        cavityRadiusVox = 10

        array = np.full(shape, 40.0, dtype=np.float32)
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        dist = np.sqrt(
            (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
        )
        array[(dist >= cavityRadiusVox) & (dist < cavityRadiusVox + 3)] = 700.0
        array[dist < cavityRadiusVox] = -900.0

        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        volumeNode.SetName("SinusSegmentationTestPhantom")
        slicer.util.updateVolumeFromArray(volumeNode, array)
        volumeNode.SetSpacing(spacing[2], spacing[1], spacing[0])  # (sx, sy, sz)

        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        # seed placed at the RAS position of the array center voxel (IJK == world here)
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        rasH = ijkToRas.MultiplyPoint((center[2], center[1], center[0], 1.0))
        fiducialNode.AddControlPoint(rasH[0], rasH[1], rasH[2])

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

        logic = SinusSegmentationLogic()
        region = SINUS_REGIONS[0]
        result = logic.segmentOneRegion(volumeNode, segmentationNode, region, fiducialNode)

        self.assertTrue(result.success, f"segmentation failed: {result.reason}")

        radiusMm = cavityRadiusVox * spacing[0]
        expectedCm3 = (4.0 / 3.0 * math.pi * radiusMm ** 3) / 1000.0
        relError = abs(result.volume_cm3 - expectedCm3) / expectedCm3
        self.assertLess(relError, 0.15, "computed volume too far from analytic sphere volume")

        self.delayDisplay(
            f"Test passed: volume_cm3={result.volume_cm3:.3f} expected_cm3={expectedCm3:.3f}"
        )
