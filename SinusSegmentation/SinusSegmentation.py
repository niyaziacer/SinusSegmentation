import csv
import logging
import os
import zipfile
from xml.sax.saxutils import escape

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


def _xlsxCellXml(colIndex, rowIndex, value):
    col = ""
    n = colIndex
    while True:
        col = chr(ord("A") + n % 26) + col
        n = n // 26 - 1
        if n < 0:
            break
    ref = f"{col}{rowIndex}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape("" if value is None else str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def writeXlsx(path, headers, rows):
    """Writes a minimal single-sheet .xlsx file using only the standard
    library (zipfile + hand-written OOXML), so no extra Python package needs
    to be installed in Slicer's Python environment."""
    sheetRowsXml = []
    allRows = [headers] + list(rows)
    for rowIndex, row in enumerate(allRows, start=1):
        cells = "".join(_xlsxCellXml(colIndex, rowIndex, value) for colIndex, value in enumerate(row))
        sheetRowsXml.append(f'<row r="{rowIndex}">{cells}</row>')

    sheetXml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheetRowsXml)}</sheetData>'
        "</worksheet>"
    )
    contentTypesXml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    rootRelsXml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbookXml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sonuclar" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbookRelsXml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", contentTypesXml)
        z.writestr("_rels/.rels", rootRelsXml)
        z.writestr("xl/workbook.xml", workbookXml)
        z.writestr("xl/_rels/workbook.xml.rels", workbookRelsXml)
        z.writestr("xl/worksheets/sheet1.xml", sheetXml)


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
        # Explicitly (re)set the scene on each node selector too: relying only
        # on the root qMRMLWidget to cascade it down has been unreliable here.
        self.ui.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentationSelector.setMRMLScene(slicer.mrmlScene)

        self.logic = SinusSegmentationLogic()

        self._buildRegionRows()
        self._buildResultsTable()

        self.ui.inputVolumeSelector.currentNodeChanged.connect(self._updateApplyButtonState)
        self.ui.applyButton.clicked.connect(self.onApplyButton)
        self.ui.exportCsvButton.clicked.connect(self.onExportCsvButton)

        if self.ui.inputVolumeSelector.currentNode() is None:
            firstVolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolume is not None:
                self.ui.inputVolumeSelector.setCurrentNode(firstVolume)

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
            seedButton.setToolTip(
                "Bir tohum eklemek için bas ve görünümde bir kez tıkla. Aynı bölgeye "
                "başka bir tohum daha eklemek istersen (örn. etmoid gibi tek parça "
                "olmayan sinüsler için) butona tekrar bas."
            )

            clearButton = qt.QPushButton("Temizle")
            clearButton.setMaximumWidth(60)

            statusLabel = qt.QLabel("tohum yok")
            statusLabel.setMinimumWidth(70)

            rowLayout.addWidget(checkbox)
            rowLayout.addWidget(seedButton)
            rowLayout.addWidget(clearButton)
            rowLayout.addWidget(statusLabel)
            layout.addWidget(rowWidget)

            seedButton.clicked.connect(lambda checked, rid=region.id: self.onPlaceSeedClicked(rid, checked))
            clearButton.clicked.connect(lambda checked=False, rid=region.id: self.onClearSeedsClicked(rid))

            self.regionWidgets[region.id] = {
                "region": region,
                "checkbox": checkbox,
                "seedButton": seedButton,
                "clearButton": clearButton,
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
            # No SetMaximumNumberOfControlPoints() call: some sinuses (ethmoid
            # in particular) are several separate air cells rather than one
            # cavity, so a region can need more than one seed.
            fiducialNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", f"seed_{regionId}"
            )
            fiducialNode.CreateDefaultDisplayNodes()
            fiducialNode.GetDisplayNode().SetSelectedColor(*widgets["region"].color_rgb)
            widgets["fiducialNode"] = fiducialNode

        self.activePlacementRegionId = regionId
        self.placementObserverTag = fiducialNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self._onSeedPlaced
        )

        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetActivePlaceNodeID(fiducialNode.GetID())
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        # One click, one point: persistent placement let stray clicks (e.g. in
        # a different slice view while navigating) silently add extra, wrong
        # seeds. To add another seed for the same region, click the button
        # again -- existing seeds for the region are kept, not cleared.
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        self._updateSeedCountLabel(regionId)

    def _onSeedPlaced(self, caller, event):
        regionId = self.activePlacementRegionId
        if regionId is None:
            return
        self._updateSeedCountLabel(regionId)
        widgets = self.regionWidgets[regionId]
        widgets["seedButton"].setChecked(False)
        self._stopSeedPlacement()

    def _updateSeedCountLabel(self, regionId):
        widgets = self.regionWidgets[regionId]
        fiducialNode = widgets["fiducialNode"]
        count = fiducialNode.GetNumberOfControlPoints() if fiducialNode is not None else 0
        widgets["statusLabel"].setText(f"{count} tohum" if count else "tıklayın...")

    def onClearSeedsClicked(self, regionId):
        widgets = self.regionWidgets[regionId]
        if self.activePlacementRegionId == regionId:
            self._stopSeedPlacement()
            widgets["seedButton"].setChecked(False)
        fiducialNode = widgets["fiducialNode"]
        if fiducialNode is not None:
            fiducialNode.RemoveAllControlPoints()
        widgets["statusLabel"].setText("tohum yok")

    def _stopSeedPlacement(self):
        if self.activePlacementRegionId is not None:
            widgets = self.regionWidgets.get(self.activePlacementRegionId)
            if widgets is not None and widgets["fiducialNode"] is not None and self.placementObserverTag is not None:
                widgets["fiducialNode"].RemoveObserver(self.placementObserverTag)
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            if interactionNode is not None:
                interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
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

            results = self.logic.segmentOneRegion(
                volumeNode,
                segmentationNode,
                region,
                fiducialNode,
                huRange=huRange,
                minSizeVoxels=minSizeVoxels,
                openingRadiusVox=openingRadiusVox,
            )
            widgets["statusLabel"].setText(self.logic.describeResults(results))
            if not any(r.success for r in results):
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
            self.parent, "Sonuçları kaydet (CSV + Excel)", "sinus_results.csv", "CSV files (*.csv)"
        )
        if not path:
            return

        nameToEnglish = {region.name_tr: region.name_en for region in SINUS_REGIONS}
        exportRows = [
            (nameToEnglish.get(name, name), volumeCm3, surfaceMm2)
            for name, volumeCm3, surfaceMm2 in self.lastResultsRows
        ]
        headers = ["sinus", "volume_cm3", "surface_mm2"]

        csvPath = path if path.lower().endswith(".csv") else path + ".csv"
        xlsxPath = os.path.splitext(csvPath)[0] + ".xlsx"

        with open(csvPath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(exportRows)

        writeXlsx(xlsxPath, headers, exportRows)

        slicer.util.infoDisplay(f"Kaydedildi:\n{csvPath}\n{xlsxPath}")


#
# SinusSegmentationLogic
#


class SinusSegmentationLogic(ScriptedLoadableModuleLogic):
    """Slicer glue: converts between MRML nodes and the pure numpy
    segmentation_core algorithm, and computes per-segment statistics."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def ijkFromFiducial(self, volumeNode, fiducialNode, pointIndex=0):
        ras = [0.0, 0.0, 0.0]
        fiducialNode.GetNthControlPointPositionWorld(pointIndex, ras)

        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)
        ijkH = rasToIjk.MultiplyPoint((ras[0], ras[1], ras[2], 1.0))
        return int(round(ijkH[0])), int(round(ijkH[1])), int(round(ijkH[2]))

    def segmentOneRegion(self, volumeNode, segmentationNode, region, fiducialNode,
                          huRange=DEFAULT_HU_RANGE,
                          minSizeVoxels=DEFAULT_MIN_SIZE_VOXELS,
                          openingRadiusVox=DEFAULT_OPENING_RADIUS_VOX):
        """Runs region growing from every seed placed for this region (a
        sinus like the ethmoid complex is several separate air cells, so it
        commonly needs more than one seed) and writes the union of all
        resulting masks into one segment. Returns the per-seed results."""
        volumeArray = slicer.util.arrayFromVolume(volumeNode)  # array axes are (k, j, i)
        spacingIJK = volumeNode.GetSpacing()  # (sx, sy, sz)
        spacingKJI = (spacingIJK[2], spacingIJK[1], spacingIJK[0])

        combinedMask = np.zeros(volumeArray.shape, dtype=bool)
        results = []
        for pointIndex in range(fiducialNode.GetNumberOfControlPoints()):
            i, j, k = self.ijkFromFiducial(volumeNode, fiducialNode, pointIndex)
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
            results.append(result)
            if result.mask.any():
                combinedMask |= result.mask

        if combinedMask.any():
            segmentation = segmentationNode.GetSegmentation()
            segmentId = segmentation.GetSegmentIdBySegmentName(region.name_tr)
            if not segmentId:
                segmentId = segmentation.AddEmptySegment(region.id, region.name_tr, region.color_rgb)
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                combinedMask.astype("uint8"), segmentationNode, segmentId, volumeNode
            )

        return results

    def describeResults(self, results):
        if not results:
            return "tohum yok"

        numOk = sum(1 for r in results if r.success)
        total = len(results)
        if numOk == total:
            return f"OK ({total} tohum)"
        if numOk == 0:
            reasons = {
                "seed_outside_bounds": "tohum hacim dışında",
                "seed_not_in_air": "tohum hava içinde değil",
                "too_small": "çok küçük",
                "possible_leak": "olası sızıntı",
            }
            firstReason = reasons.get(results[0].reason, results[0].reason or "başarısız")
            return f"başarısız (0/{total}: {firstReason})"
        return f"kısmen OK ({numOk}/{total})"

    def computeStatistics(self, segmentationNode):
        """Returns a list of (segmentName, volume_cm3, surface_mm2) tuples,
        using Slicer's built-in SegmentStatistics module rather than a
        hand-rolled marching-cubes pass."""
        import SegmentStatistics

        # The closed-surface representation is what the surface-area plugin
        # measures; make sure it actually exists rather than assuming the
        # plugin will create it on demand.
        if not segmentationNode.GetSegmentation().ContainsRepresentation("Closed surface"):
            segmentationNode.CreateClosedSurfaceRepresentation()

        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
        # Plugin-enable parameter names have varied across Slicer versions;
        # try every spelling we know of rather than relying on one -- the
        # lookup below searches by stat-key suffix instead of exact key, so
        # it works even if a particular plugin is already enabled by default.
        for pluginName in (
            "LabelmapSegmentStatistics", "LabelmapSegmentStatisticsPlugin",
            "ClosedSurfaceSegmentStatistics", "ClosedSurfaceSegmentStatisticsPlugin",
        ):
            try:
                segStatLogic.getParameterNode().SetParameter(f"{pluginName}.enabled", str(True))
            except Exception:
                pass
        segStatLogic.computeStatistics()
        stats = segStatLogic.getStatistics()

        rows = []
        for segmentId in stats.get("SegmentIDs", []):
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            name = segment.GetName() if segment else segmentId
            volumeCm3 = self._findStatBySuffix(stats, segmentId, "volume_cm3")
            surfaceMm2 = self._findStatBySuffix(stats, segmentId, "surface_mm2")
            rows.append((name, volumeCm3, surfaceMm2))

        if rows and any(v is None or s is None for _, v, s in rows):
            logging.warning(
                "SinusSegmentation: some volume/surface stats did not match; "
                "available SegmentStatistics keys: %s",
                sorted({k[1] for k in stats if isinstance(k, tuple)}),
            )
        return rows

    @staticmethod
    def _findStatBySuffix(stats, segmentId, keySuffix):
        """stats is keyed by (segmentId, statKey) tuples, but the exact
        statKey prefix (which plugin produced it) isn't guaranteed across
        Slicer versions, so match on suffix instead of the full key."""
        for key, value in stats.items():
            if isinstance(key, tuple) and len(key) == 2 and key[0] == segmentId and key[1].endswith(keySuffix):
                return value
        return None


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
        results = logic.segmentOneRegion(volumeNode, segmentationNode, region, fiducialNode)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result.success, f"segmentation failed: {result.reason}")

        radiusMm = cavityRadiusVox * spacing[0]
        expectedCm3 = (4.0 / 3.0 * math.pi * radiusMm ** 3) / 1000.0
        relError = abs(result.volume_cm3 - expectedCm3) / expectedCm3
        self.assertLess(relError, 0.15, "computed volume too far from analytic sphere volume")

        self.delayDisplay(
            f"Test passed: volume_cm3={result.volume_cm3:.3f} expected_cm3={expectedCm3:.3f}"
        )
