from __future__ import print_function

import os
import sys
from PyQt4 import (QtCore, QtGui)
from PyQt4.QtCore import Qt
import matplotlib.cm as cm
from PIL import Image
# from PIL import ImageQt
import PIL
import csv
import multiprocessing as mp
import time
from scipy.misc import imsave
from skimage import exposure

import numpy as np

import matplotlib as mpl
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as
                    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import psutil

import load_timepix
import dpc_kernel as dpc
import pyspecfile

from databroker import DataBroker
from scan import Scan

get_save_filename = QtGui.QFileDialog.getSaveFileName


SOLVERS = ['Nelder-Mead',
           'Powell',
           'CG',
           'BFGS',
           'Newton-CG',
           'Anneal',
           'L-BFGS-B',
           'TNC',
           'COBYLA',
           'SLS   QP',
           'dogleg',
           'trust-ncg',
           ]

TYPES = ['TIFF',
         'Timepix TIFF',
         'ASCII',
         'FileStore',
         ]

roi_x1 = 0
roi_x2 = 0
roi_y1 = 0
roi_y2 = 0
a = None
gx = None
gy = None
phi = None

CMAP_PREVIEW_PATH = os.path.join(os.path.dirname(__file__), '.cmap_previews')


def load_image_pil(path):
    """
    Read images using the PIL lib
    """
    file = Image.open(str(path))  # 'I;16B'
    return np.array(file.getdata()).reshape(file.size[::-1])


def load_image_ascii(path):
    """
    Read ASCII images using the csv lib
    """
    delimiter = '\t'
    data = []
    for row in csv.reader(open(path), delimiter=delimiter):
        data.append(row[:-1])
    img = np.array(data).astype(np.double)
    return img


def brush_to_color_tuple(brush):
    r, g, b, a = brush.color().getRgbF()
    return (r, g, b)


class DPCThread(QtCore.QThread):
    def __init__(self, canvas, pool=None, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.canvas = canvas
        self.pool = pool

    update_signal = QtCore.pyqtSignal(object, object, object, object)

    def run(self):
        print('DPC thread started')
        main = DPCWindow.instance
        try:
            ret = dpc.main(pool=self.pool,
                           display_fcn=lambda *args: self.update_signal.emit(*args),
                           load_image=main.load_image,
                           **self.dpc_settings)
            print('DPC finished')
            global a
            global gx
            global gy
            global phi
            a, gx, gy, phi = ret

            main.a, main.gx, main.gy, main.phi = a, gx, gy, phi
            main.line_btn.setEnabled(True)
            main.reverse_x.setEnabled(True)
            main.reverse_y.setEnabled(True)
            main.swap_xy.setEnabled(True)
            main.save_result.setEnabled(True)
            main.hanging_opt.setEnabled(True)
            main.random_processing_opt.setEnabled(True)
            main.pyramid_scan.setEnabled(True)
            # main.direction_btn.setEnabled(True)
            # main.removal_btn.setEnabled(True)
            # main.confirm_btn.setEnabled(True)
        finally:
            main.set_running(False)


class MplCanvas(FigureCanvas):
    """
    Canvas which allows us to use matplotlib with pyqt4
    """
    def __init__(self, fig=None, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        # We want the axes cleared every time plot() is called
        self.axes = fig.add_subplot(1, 1, 1)
        self.axes.hold(False)

        FigureCanvas.__init__(self, fig)

        # self.figure
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self._title = ''
        self.title_font = {'family': 'serif', 'fontsize': 10}
        self._title_size = 0
        self.figure.subplots_adjust(top=0.95, bottom=0.15)

        window_brush = self.window().palette().window()
        fig.set_facecolor(brush_to_color_tuple(window_brush))
        fig.set_edgecolor(brush_to_color_tuple(window_brush))
        self._active = False

    def _get_title(self):
        return self._title

    def _set_title(self, title):
        self._title = title
        if self.axes:
            self.axes.set_title(title, fontdict=self.title_font)
            #bbox = t.get_window_extent()
            #bbox = bbox.inverse_transformed(self.figure.transFigure)
            #self._title_size = bbox.height
            #self.figure.subplots_adjust(top=1.0 - self._title_size)

    title = property(_get_title, _set_title)


class Label(QtGui.QLabel):
    def __init__(self, parent = None):
        super(Label, self).__init__(parent)
        self.rubberBand = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle, self)
        self.origin = QtCore.QPoint()

    def mousePressEvent(self, event):
        global roi_x1
        global roi_y1
        self.rubberBand.hide()
        if event.button() == Qt.LeftButton:
            self.origin = QtCore.QPoint(event.pos())
            self.rubberBand.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.rubberBand.show()
            roi_x1 = event.pos().x()
            roi_y1 = event.pos().y()

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.NoButton:
            pos = event.pos()
        if not self.origin.isNull():
            self.rubberBand.setGeometry(QtCore.QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        global roi_x2
        global roi_y2
        roi_x2 = event.pos().x()
        roi_y2 = event.pos().y()
        main = DPCWindow.instance
        if ((roi_x1, roi_y1)!=(roi_x2, roi_y2)):
            main.roi_x1_widget.setValue(roi_x1)
            main.roi_y1_widget.setValue(roi_y1)
            main.roi_x2_widget.setValue(roi_x2)
            main.roi_y2_widget.setValue(roi_y2)
        else:
            if main.bad_flag != 0:
                main.bad_pixels_widget.addItem('%d, %d' %
                (event.pos().x(), event.pos().y()))
                self.rubberBand.show()


class paintLabel(QtGui.QLabel):
    def __init__(self, parent = None):
        super(paintLabel, self).__init__(parent)

    def paintEvent(self, event):
        super(paintLabel, self).paintEvent(event)
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawLine(event, qp)
        qp.end()

    def drawLine(self, event, qp):
        size = self.size()
        pen = QtGui.QPen(QtCore.Qt.red)
        qp.setPen(pen)
        qp.drawLine(size.width()/2, 0, size.width()/2, size.height()-1)
        qp.drawLine(size.width()/2 - 1, 0, size.width()/2 - 1, size.height()-1)
        qp.drawLine(0, size.height()/2, size.width()-1, size.height()/2)
        qp.drawLine(0, size.height()/2-1, size.width()-1, size.height()/2-1)

        pen.setStyle(QtCore.Qt.DashLine)
        pen.setColor(QtCore.Qt.black)
        qp.setPen(pen)
        qp.drawLine(0, 0, size.width()-1, 0)
        qp.drawLine(0, size.height()-1, size.width()-1, size.height()-1)
        qp.drawLine(0, 0, 0, size.height()-1)
        qp.drawLine(size.width()-1, 0, size.width()-1, size.height()-1)


class DPCWindow(QtGui.QMainWindow):
    CM_DEFAULT = 'gray'
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        DPCWindow.instance = self

        self.bin_num = 2**16
        self._thread = None
        self.ion_data = None
        self.bad_flag = 0
        self.direction = 1 # 1 for horizontal and -1 for vertical
        self.crop_x0 = None
        self.crop_x1 = None
        self.crop_y0 = None
        self.crop_y1 = None
        self.set_roi_enabled = False
        self.his_enabled = False

        self.gx, self.gy, self.phi, self.a = None, None, None, None
        self.file_widget = QtGui.QLineEdit('Chromosome_9_%05d.tif')
        self.file_widget.setFixedWidth(350)
        self.focus_widget = QtGui.QDoubleSpinBox()

        self.dx_widget = QtGui.QDoubleSpinBox()
        self.dy_widget = QtGui.QDoubleSpinBox()
        self.pixel_widget = QtGui.QDoubleSpinBox()
        self.energy_widget = QtGui.QDoubleSpinBox()
        self.rows_widget = QtGui.QSpinBox()
        self.cols_widget = QtGui.QSpinBox()
        self.mosaic_x_widget = QtGui.QSpinBox()
        self.mosaic_y_widget = QtGui.QSpinBox()
        self.roi_x1_widget = QtGui.QSpinBox()
        self.roi_x2_widget = QtGui.QSpinBox()
        self.roi_y1_widget = QtGui.QSpinBox()
        self.roi_y2_widget = QtGui.QSpinBox()
        self.strap_start = QtGui.QSpinBox()
        self.strap_end = QtGui.QSpinBox()
        self.first_widget = QtGui.QSpinBox()
        self.processes_widget = QtGui.QSpinBox()
        self.processes_widget.setMinimum(1)
        self.processes_widget.setValue(psutil.cpu_count())
        self.processes_widget.setMaximum(psutil.cpu_count())

        self.solver_widget = QtGui.QComboBox()
        for solver in SOLVERS:
            self.solver_widget.addItem(solver)

        self.start_widget = QtGui.QPushButton('Start')
        self.stop_widget = QtGui.QPushButton('Stop')
        self.save_widget = QtGui.QPushButton('Save')
        self.scan_button = QtGui.QPushButton('Load')

        self.color_map = QtGui.QComboBox()
        self.update_color_maps()
        self.color_map.currentIndexChanged.connect(self._set_color_map)
        self._color_map = mpl.cm.get_cmap(self.CM_DEFAULT)

        self.ref_color_map = QtGui.QComboBox()
        self.update_ref_color_maps()
        self.ref_color_map.currentIndexChanged.connect(self._set_ref_color_map)
        self._ref_color_map = mpl.cm.get_cmap(self.CM_DEFAULT)

        self.start_widget.clicked.connect(self.start)
        self.stop_widget.clicked.connect(self.stop)
        self.save_widget.clicked.connect(self.save)
        self.scan_button.clicked.connect(self.load_from_spec_scan)

        self.load_image = load_timepix.load

        def format_coord(x, y):
            col = int(x+0.5)
            row = int(y+0.5)
            if row>=0 and row<self.roi_img.shape[0] and col>=0 and col<self.roi_img.shape[1]:
                z = self.roi_img[row, col]
                return 'x=%1.4f   y=%1.4f   v=%1.4f'%(x, y, z)
            else:
                return 'x=%1.4f   y=%1.4f'%(x, y)

        self.rect = Rectangle((0, 0), 0, 0, alpha=0.3, facecolor='gray',
                              edgecolor='red', linewidth=2)
        self.ref_fig = plt.figure()
        #self.ref_canvas = MplCanvas(self.ref_fig, width=8, height=10, dpi=50)
        self.ref_canvas = FigureCanvas(self.ref_fig)
        self.ref_fig.subplots_adjust(top=0.99, left=0.01, right=0.99, bottom=0.04)
        self.ref_fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.ref_fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.ref_fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax = self.ref_fig.add_subplot(111)
        self.ax.format_coord = format_coord
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.draw()
        self.ref_toolbar = NavigationToolbar(self.ref_canvas, self)

        self.his_btn = QtGui.QPushButton('Equalize')
        self.his_btn.setCheckable(True)
        self.his_btn.clicked[bool].connect(self.histgramEqua)
        self.roi_btn = QtGui.QPushButton('Set ROI')
        self.roi_btn.setCheckable(True)
        self.roi_btn.clicked[bool].connect(self.set_roi_enable)
        self.bri_btn = QtGui.QPushButton('Brightest')
        self.bri_btn.clicked.connect(self.select_bri_pixels)
        self.bad_btn = QtGui.QPushButton('Pick')
        self.bad_btn.setCheckable(True)
        self.bad_btn.clicked[bool].connect(self.bad_enable)

        self.line_btn = QtGui.QPushButton('Add')
        self.line_btn.setEnabled(False)
        self.line_btn.clicked.connect(self.add_strap)
        self.direction_btn = QtGui.QPushButton(u'\N{CLOCKWISE OPEN CIRCLE ARROW} 90\N{DEGREE SIGN}')
        self.direction_btn.clicked.connect(self.change_direction)
        self.direction_btn.setEnabled(False)
        self.removal_btn = QtGui.QPushButton('Remove')
        self.removal_btn.clicked.connect(self.remove_background)
        self.removal_btn.setEnabled(False)
        self.confirm_btn = QtGui.QPushButton('Apply')
        self.confirm_btn.clicked.connect(self.confirm)
        self.confirm_btn.setEnabled(False)
        self.hide_btn = QtGui.QPushButton("View && set")
        self.hide_btn.setCheckable(True)
        self.hide_btn.clicked.connect(self.hide_ref)

        self.ok_btn = QtGui.QPushButton('OK')
        self.ok_btn.clicked.connect(self.crop_ok)
        self.cancel_btn = QtGui.QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.crop_cancel)

        # Setting widget (QGridLayout) in the bottom of reference image
        self.min_lbl= QtGui.QLabel('Min')
        self.max_lbl = QtGui.QLabel('Max')
        self.min_box = QtGui.QSpinBox()
        self.max_box = QtGui.QSpinBox()
        self.min_box.setMaximum(self.bin_num)
        self.min_box.setMinimum(0)
        self.max_box.setMaximum(self.bin_num)
        self.max_box.setMinimum(0)
        self.rescale_intensity_btn= QtGui.QPushButton('Apply')
        self.rescale_intensity_btn.clicked.connect(self.rescale_intensity)

        self.badPixelGbox = QtGui.QGroupBox("Bad pixels")
        self.badPixelGridLayout = QtGui.QGridLayout()
        self.badPixelGbox.setLayout(self.badPixelGridLayout)
        self.bad_pixels_widget = QtGui.QListWidget()
        # Set the minimum height of the qlistwidget as 1 so that the qlistwidget
        # is always as as high as its two side buttons
        self.bad_pixels_widget.setMinimumHeight(1)
        self.bad_pixels_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.bad_pixels_widget.customContextMenuRequested.connect(self._bad_pixels_menu)
        self.badPixelGridLayout.addWidget(self.bri_btn, 0, 0)
        self.badPixelGridLayout.addWidget(self.bad_btn, 1, 0)
        self.badPixelGridLayout.addWidget(self.bad_pixels_widget, 0, 1, 2, 1)

        def ref_close(event):
            self.hide_btn.setChecked(False)
        self.ref_grid = QtGui.QGridLayout()
        self.ref_widget = QtGui.QWidget()
        self.ref_widget.closeEvent = ref_close
        self.ref_widget.setLayout(self.ref_grid)
        self.ref_grid.addWidget(self.ref_canvas, 0, 0, 1, 6)
        self.ref_grid.addWidget(self.ref_toolbar, 1, 0, 1, 6)
        self.ref_grid.addWidget(self.badPixelGbox, 2, 0, 3, 1)
        self.ref_grid.addWidget(self.ref_color_map, 2, 1, 1, 5)
        self.ref_grid.addWidget(self.his_btn, 3, 1, 1, 2)
        self.ref_grid.addWidget(self.roi_btn, 3, 3, 1, 2)
        self.ref_grid.addWidget(self.min_lbl, 4, 1)
        self.ref_grid.addWidget(self.min_box, 4, 2)
        self.ref_grid.addWidget(self.max_lbl, 4, 3)
        self.ref_grid.addWidget(self.max_box, 4, 4)
        self.ref_grid.addWidget(self.rescale_intensity_btn, 4, 5)

        self.file_format_btn = QtGui.QPushButton('Select')
        #self.file_format_btn.setStyle(WinLayout)
        self.file_format_btn.clicked.connect(self.select_path)

        """
        QGroupBox implementation for image settings
        """
        self.imageSettingGbox = QtGui.QGroupBox("Image settings")
        self.imageSettingGridLayout = QtGui.QGridLayout()
        self.imageSettingGbox.setLayout(self.imageSettingGridLayout)
        self.scan_number_lbl = QtGui.QLabel('Scan number')
        self.roi_x1_lbl = QtGui.QLabel('ROI X1')
        self.roi_x2_lbl = QtGui.QLabel('ROI X2')
        self.roi_y1_lbl = QtGui.QLabel('ROI Y1')
        self.roi_y2_lbl = QtGui.QLabel('ROI Y2')
        self.img_type_lbl = QtGui.QLabel('Image type')
        self.pixel_size_lbl = QtGui.QLabel('Pixel size (um)')
        self.file_name_lbl = QtGui.QLabel('File name')
        self.first_img_num_lbl = QtGui.QLabel('First image number')
        self.scan_info_lbl = QtGui.QLabel('')
        self.scan_info_lbl.setWordWrap(True)
        self.select_ref_btn = QtGui.QPushButton('Select the reference')
        self.select_ref_btn.clicked.connect(self.select_ref_img)
        self.img_type_combobox = QtGui.QComboBox()
        for types in TYPES:
            self.img_type_combobox.addItem(types)
        self.img_type_combobox.currentIndexChanged.connect(self.load_img_method)
        self.first_img_as_ref_checkbox = QtGui.QCheckBox("Use as the reference image")
        self.first_img_as_ref_checkbox.stateChanged.connect(self.first_equal_ref)
        self.use_scan_number_cb = QtGui.QCheckBox("Read from metadatastore")
        self.use_scan_number_cb.toggled.connect(self._use_scan_number_clicked)
        self.filestore_key_combo = QtGui.QComboBox()
        self.filestore_key_combo.currentIndexChanged.connect(self._filestore_key_changed)
        self.load_scan_btn = QtGui.QPushButton('Load')
        self.load_scan_btn.clicked.connect(self.load_scan_from_mds)

        self.ref_image_path_QLineEdit = QtGui.QLineEdit('reference image')
        self.ref_image_path_QLineEdit.setFixedWidth(350)
        self.scan_number_text = QtGui.QLineEdit('3449')

        row = 0
        self.imageSettingGridLayout.addWidget(self.scan_number_lbl, row, 0)
        self.imageSettingGridLayout.addWidget(self.scan_number_text, row, 1)
        self.imageSettingGridLayout.addWidget(self.load_scan_btn, row, 2)
        self.imageSettingGridLayout.addWidget(self.use_scan_number_cb, row, 3)

        row += 1
        self.imageSettingGridLayout.addWidget(self.filestore_key_combo, row, 0)
        self.imageSettingGridLayout.addWidget(self.scan_info_lbl, row, 1, 1, 3)

        row += 1
        self.imageSettingGridLayout.addWidget(self.img_type_lbl, row, 0)
        self.imageSettingGridLayout.addWidget(self.img_type_combobox, row, 1)
        self.imageSettingGridLayout.addWidget(self.pixel_size_lbl, row, 2)
        self.imageSettingGridLayout.addWidget(self.pixel_widget, row, 3)

        row += 1
        self.imageSettingGridLayout.addWidget(self.file_name_lbl, row, 0)
        self.imageSettingGridLayout.addWidget(self.file_widget, row, 1, 1, 3)
        self.imageSettingGridLayout.addWidget(self.file_format_btn, row, 4)

        row += 1
        self.imageSettingGridLayout.addWidget(self.first_img_num_lbl, row, 0)
        self.imageSettingGridLayout.addWidget(self.first_widget, row, 1)
        self.imageSettingGridLayout.addWidget(self.first_img_as_ref_checkbox, row, 2, 1, 2)

        row += 1
        self.imageSettingGridLayout.addWidget(self.select_ref_btn, row, 0)
        self.imageSettingGridLayout.addWidget(self.ref_image_path_QLineEdit, row, 1, 1, 3)

        row += 1
        self.imageSettingGridLayout.addWidget(self.roi_x1_lbl, row, 0)
        self.imageSettingGridLayout.addWidget(self.roi_x1_widget, row, 1)
        self.imageSettingGridLayout.addWidget(self.roi_x2_lbl, row, 2)
        self.imageSettingGridLayout.addWidget(self.roi_x2_widget, row, 3)
        self.imageSettingGridLayout.addWidget(self.hide_btn, row, 4)

        row += 1
        self.imageSettingGridLayout.addWidget(self.roi_y1_lbl, row, 0)
        self.imageSettingGridLayout.addWidget(self.roi_y1_widget, row, 1)
        self.imageSettingGridLayout.addWidget(self.roi_y2_lbl, row, 2)
        self.imageSettingGridLayout.addWidget(self.roi_y2_widget, row, 3)

        # QGroupBox implementation for experiment parameters
        self.experimentParaGbox = QtGui.QGroupBox("Experiment parameters")
        self.experimentParaGridLayout = QtGui.QGridLayout()
        self.experimentParaGbox.setLayout(self.experimentParaGridLayout)
        self.energy_lbl = QtGui.QLabel('Energy (keV)')
        self.detector_sample_lbl = QtGui.QLabel('Detector-sample distance (m)')
        self.x_step_size_lbl = QtGui.QLabel('X step size (um)')
        self.y_step_size_lbl = QtGui.QLabel('Y step size (um)')
        self.x_steps_number_lbl = QtGui.QLabel('Columns (x)')
        self.y_steps_number_lbl = QtGui.QLabel('Rows (y)')
        self.mosaic_x_size_lbl = QtGui.QLabel('Mosaic column number')
        self.mosaic_y_size_lbl = QtGui.QLabel('Mosaic row number')
        self.experimentParaGridLayout.addWidget(self.energy_lbl, 0, 0)
        self.experimentParaGridLayout.addWidget(self.energy_widget, 0, 1)
        self.experimentParaGridLayout.addWidget(self.detector_sample_lbl, 0, 2)
        self.experimentParaGridLayout.addWidget(self.focus_widget, 0, 3)
        self.experimentParaGridLayout.addWidget(self.x_step_size_lbl, 1, 0)
        self.experimentParaGridLayout.addWidget(self.dx_widget, 1, 1)
        self.experimentParaGridLayout.addWidget(self.y_step_size_lbl, 1, 2)
        self.experimentParaGridLayout.addWidget(self.dy_widget, 1, 3)
        self.experimentParaGridLayout.addWidget(self.x_steps_number_lbl, 2, 0)
        self.experimentParaGridLayout.addWidget(self.cols_widget, 2, 1)
        self.experimentParaGridLayout.addWidget(self.y_steps_number_lbl, 2, 2)
        self.experimentParaGridLayout.addWidget(self.rows_widget, 2, 3)
        self.experimentParaGridLayout.addWidget(self.mosaic_x_size_lbl, 3, 0)
        self.experimentParaGridLayout.addWidget(self.mosaic_x_widget, 3, 1)
        self.experimentParaGridLayout.addWidget(self.mosaic_y_size_lbl, 3, 2)
        self.experimentParaGridLayout.addWidget(self.mosaic_y_widget, 3, 3)

        """
        QGroupBox implementation for computation parameters
        """
        self.computationParaGbox = QtGui.QGroupBox("Computation parameters")
        self.computationParaGridLayout = QtGui.QGridLayout()
        self.computationParaGbox.setLayout(self.computationParaGridLayout)
        self.solver_method_lbl = QtGui.QLabel('Solver method')
        self.processes_lbl = QtGui.QLabel('Processes')
        self.random_processing_checkbox = QtGui.QCheckBox("Random mode")
        self.hanging_checkbox = QtGui.QCheckBox("Hanging mode")
        self.computationParaGridLayout.addWidget(self.solver_method_lbl, 0, 0)
        self.computationParaGridLayout.addWidget(self.solver_widget, 0, 1)
        self.computationParaGridLayout.addWidget(self.processes_lbl, 0, 2)
        self.computationParaGridLayout.addWidget(self.processes_widget, 0, 3)
        #self.computationParaGridLayout.addWidget(self.random_processing_checkbox, 1, 0)
        #self.computationParaGridLayout.addWidget(self.hanging_checkbox, 1, 1)
        self.computationParaGridLayout.addWidget(self.start_widget, 0, 4)
        self.computationParaGridLayout.addWidget(self.stop_widget, 0, 5)

        self.background_remove_qbox = QtGui.QGroupBox("Remove background")
        self.background_remove_layout = QtGui.QGridLayout()
        self.background_remove_qbox.setLayout(self.background_remove_layout)
        self.strap_start_label = QtGui.QLabel('Start')
        self.strap_end_label = QtGui.QLabel('End')
        self.background_remove_layout.addWidget(self.strap_start_label, 0,0)
        self.background_remove_layout.addWidget(self.strap_start, 0, 1)
        self.background_remove_layout.addWidget(self.strap_end_label, 0, 2)
        self.background_remove_layout.addWidget(self.strap_end, 0, 3)
        self.background_remove_layout.addWidget(self.line_btn, 0, 4)
        self.background_remove_layout.addWidget(self.direction_btn, 0, 5)
        self.background_remove_layout.addWidget(self.removal_btn, 0, 6)
        self.background_remove_layout.addWidget(self.confirm_btn, 0, 7)

        self.canvas = MplCanvas(width=10, height=12, dpi=50)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.image_visualization_qbox = QtGui.QGroupBox("Image visualization")
        self.image_visualization_layout = QtGui.QGridLayout()
        self.image_visualization_qbox.setLayout(self.image_visualization_layout)
        self.image_visualization_layout.addWidget(self.toolbar, 0, 0)
        self.image_visualization_layout.addWidget(self.color_map, 0, 1)

        self.canvas_QGridLayout = QtGui.QGridLayout()
        self.canvas_widget = QtGui.QWidget()
        self.canvas_widget.setLayout(self.canvas_QGridLayout)
        self.canvas_QGridLayout.addWidget(self.canvas, 0, 0, 1, 2)
        self.canvas_QGridLayout.addWidget(self.image_visualization_qbox, 1, 0)
        self.canvas_QGridLayout.addWidget(self.background_remove_qbox, 1, 1)

        self.crop_widget = QtGui.QWidget()
        self.crop_layout = QtGui.QGridLayout()
        self.crop_widget.setLayout(self.crop_layout)
        self.crop_canvas = MplCanvas(width=8, height=8, dpi=50)
        self.crop_fig = self.crop_canvas.figure
        self.crop_fig.subplots_adjust(top=0.95, left=0.05, right=0.95, bottom=0.05)
        self.crop_ax = self.crop_fig.add_subplot(111)
        self.crop_ax.hold(False)
        self.crop_layout.addWidget(self.crop_canvas, 0, 0, 1, 2)
        self.crop_layout.addWidget(self.ok_btn, 1, 0)
        self.crop_layout.addWidget(self.cancel_btn, 1, 1)

        self.last_path = ''

        self.main_grid = QtGui.QGridLayout()
        self.main_widget = QtGui.QWidget()
        self.main_widget.setLayout(self.main_grid)
        self.main_grid.addWidget(self.imageSettingGbox, 0, 0)
        self.main_grid.addWidget(self.experimentParaGbox, 1, 0)
        self.main_grid.addWidget(self.computationParaGbox, 2, 0)

        # Add menu
        self.menu = self.menuBar()
        self.save_result = QtGui.QAction('Save result', self)
        self.save_result.setEnabled(False)
        self.save_result.triggered.connect(self.save_file)
        self.reverse_x = QtGui.QAction('Reverse gx', self, checkable=True)
        self.reverse_x.triggered.connect(self.reverse_gx)
        self.reverse_x.setEnabled(False)
        self.reverse_y = QtGui.QAction('Reverse gy', self, checkable=True)
        self.reverse_y.triggered.connect(self.reverse_gy)
        self.reverse_y.setEnabled(False)
        self.swap_xy = QtGui.QAction('Swap x/y', self, checkable=True)
        self.swap_xy.triggered.connect(self.swap_x_y)
        self.swap_xy.setEnabled(False)
        self.random_processing_opt = QtGui.QAction('Random mode', self, checkable=True)
        self.hanging_opt = QtGui.QAction('Hanging mode', self, checkable=True)
        self.pyramid_scan = QtGui.QAction('Pyramid scan', self, checkable=True)

        file_menu = self.menu.addMenu('File')
        file_menu.addAction(self.save_result)
        option_menu = self.menu.addMenu('Option')
        option_menu.addAction(self.reverse_x)
        option_menu.addAction(self.reverse_y)
        option_menu.addAction(self.swap_xy)
        option_menu.addAction(self.random_processing_opt)
        option_menu.addAction(self.hanging_opt)
        option_menu.addAction(self.pyramid_scan)

        self.setCentralWidget(self.main_widget)
        self.setWindowTitle('DPC')

        #QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Plastique'))
        #QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('cde'))

        self._settings = {
            'file_format': [lambda: self.file_format, lambda value: self.file_widget.setText(value)],
            'dx': [lambda: self.dx, lambda value: setattr(self, 'dx', value)],
            'dy': [lambda: self.dy, lambda value: setattr(self, 'dy', value)],

            'x1': [lambda: self.roi_x1, lambda value: self.roi_x1_widget.setValue(int(value))],
            'y1': [lambda: self.roi_y1, lambda value: self.roi_y1_widget.setValue(int(value))],
            'x2': [lambda: self.roi_x2, lambda value: self.roi_x2_widget.setValue(int(value))],
            'y2': [lambda: self.roi_y2, lambda value: self.roi_y2_widget.setValue(int(value))],

            'pixel_size': [lambda: self.pixel_size, lambda value: self.pixel_widget.setValue(float(value))],
            'focus_to_det': [lambda: self.focus, lambda value: self.focus_widget.setValue(float(value))],
            'energy': [lambda: self.energy, lambda value: self.energy_widget.setValue(float(value))],

            'rows': [lambda: self.rows, lambda value: self.rows_widget.setValue(int(value))],
            'cols': [lambda: self.cols, lambda value: self.cols_widget.setValue(int(value))],
            'mosaic_y': [lambda: self.mosaic_y, lambda value: self.mosaic_y_widget.setValue(int(value))],
            'mosaic_x': [lambda: self.mosaic_x, lambda value: self.mosaic_x_widget.setValue(int(value))],
            'swap':[lambda: self.swap, lambda value: self.swap_xy.setChecked(int(value)+1)],
            'reverse_x':[lambda: self.re_x, lambda value: self.reverse_x.setChecked(int(value)-1)],
            'reverse_y':[lambda: self.re_y, lambda value: self.reverse_y.setChecked(int(value)-1)],
            'random': [lambda: self.random, lambda value: self.random_processing_opt.setChecked(int(value)+1)],
            'pyramid': [lambda: self.pyramid, lambda value: self.pyramid_scan.setChecked(int(value)+1)],
            'hang': [lambda: self.hang, lambda value: self.hanging_opt.setChecked(int(value)+1)],
            'ref_image': [lambda: self.ref_image, lambda value: self.ref_image_path_QLineEdit.setText(str(value))],
            'first_image': [lambda: self.first_image, lambda value: self.first_widget.setValue(int(value))],
            #'processes': [lambda: self.processes, lambda value: self.processes_widget.setValue(int(value))],
            'bad_pixels': [lambda: self.bad_pixels, lambda value: self.set_bad_pixels(value)],
            'solver': [lambda: self.solver, lambda value: self.set_solver(value)],
            'last_path': [lambda: self.last_path, lambda value: setattr(self, 'last_path', value)],

            'scan_number': [lambda: self.scan_number, lambda value: setattr(self, 'scan_number', value)],
            'use_mds': [lambda: self.use_mds, lambda value: setattr(self, 'use_mds', value)],
            'filestore_key': [lambda: self.filestore_key, lambda value: setattr(self, 'filestore_key', value)],
            #'color_map': [lambda: self._color_map, lambda value: setattr(self, 'last_path', value)],
            }

        for w in [self.pixel_widget, self.focus_widget, self.energy_widget,
                  self.dx_widget, self.dy_widget, self.rows_widget, self.cols_widget,
                  self.roi_x1_widget, self.roi_x2_widget, self.roi_y1_widget, self.roi_y2_widget,
                  self.first_widget, self.mosaic_x_widget, self.mosaic_y_widget,
                  self.strap_start, self.strap_end,
                  ]:
            w.setMinimum(0)
            w.setMaximum(int(2 ** 31 - 1))
            try:
                w.setDecimals(3)
            except:
                pass

        for w in [self.strap_start, self.strap_end]:
            w.setMinimum(0)
            w.setMaximum(9999)

        self.load_settings()

    def _use_scan_number_clicked(self, checked):
        self.use_mds = checked
        self.file_widget.setEnabled(not self.use_mds)
        self.filestore_key_combo.setVisible(self.use_mds)
        self.scan_info_lbl.setVisible(self.use_mds)

        if self.use_mds:
            self.img_type_combobox.setCurrentIndex(TYPES.index('FileStore'))

        self.img_type_combobox.setEnabled(not self.use_mds)

    @property
    def use_mds(self):
        return bool(self.use_scan_number_cb.isChecked())

    @use_mds.setter
    def use_mds(self, checked):
        self.use_scan_number_cb.setChecked(bool(checked))

    @property
    def filestore_key(self):
        return str(self.filestore_key_combo.currentText())

    @filestore_key.setter
    def filestore_key(self, key):
        keys = list(sorted(self.scan.filestore_keys))
        self.filestore_key_combo.setCurrentIndex(keys.index(key))

    def _load_scan_from_mds(self, scan_id):
        hdrs = DataBroker.find_headers(scan_id=scan_id)
        if len(hdrs) > 1:
            # TODO selection widget
            print('Multiple headers found...')

        hdr = hdrs[0]
        self.scan = Scan(hdr)
        selected = self.filestore_key
        self.filestore_key_combo.clear()
        for i, key in enumerate(sorted(self.scan.filestore_keys)):
            self.filestore_key_combo.addItem(key)
            if key == selected:
                self.filestore_key_combo.setCurrentIndex(i)

        self.use_mds = True

        if self.scan.dimensions is None or len(self.scan.dimensions) == 0:
            return

        scan_range = self.scan.range
        print('Scan dimensions', self.scan.dimensions)
        print('Scan range:', scan_range)

        if len(self.scan.dimensions) == 1:
            nx, ny = self.scan.dimensions[0], 1
            if scan_range is not None:
                self.dx = np.diff(scan_range[0]) / nx
                self.dy = 0.0
        else:
            nx, ny = self.scan.dimensions
            if scan_range is not None:
                self.dx = np.diff(scan_range[0]) / nx
                self.dy = np.diff(scan_range[1]) / ny

        self.cols = nx
        self.rows = ny

        command = self.scan.command
        self.scan_info_lbl.setText(command)

    def load_scan_from_mds(self):
        return self._load_scan_from_mds(self.scan_number)

    @property
    def scan_number(self):
        return int(self.scan_number_text.text())

    @scan_number.setter
    def scan_number(self, value):
        self.scan_number_text.setText(str(value))

    def _filestore_key_changed(self, event):
        key = self.filestore_key
        if self.scan is not None:
            self.scan.key = key
            print('MDS key set:', key)
            iter_ = iter(self.scan)
            ref_image = next(iter_)
            self.ref_image_path_QLineEdit.setText(ref_image)

    def on_press(self, event):
        if event.inaxes:
            self.crop_x0 = event.xdata
            self.crop_y0 = event.ydata

    def on_release(self, event):
        self.crop_x1 = event.xdata
        self.crop_y1 = event.ydata
        if event.inaxes:
            if (self.crop_x0, self.crop_y0) == (self.crop_x1, self.crop_y1):
                if self.bad_flag:
                    self.bad_pixels_widget.addItem('%d, %d' %
                                                   (int(round(self.crop_x1)),
                                                    int(round(self.crop_y1))))
            elif self.set_roi_enabled:
                if self.his_enabled:
                    roi_crop = self.roi_img_equ[int(round(self.crop_y0)):
                                                int(round(self.crop_y1)),
                                                int(round(self.crop_x0)):
                                                int(round(self.crop_x1))]
                else:
                    roi_crop = self.roi_img[int(round(self.crop_y0)):
                                            int(round(self.crop_y1)),
                                            int(round(self.crop_x0)):
                                            int(round(self.crop_x1))]
                self.crop_ax.imshow(roi_crop,
                                    interpolation='nearest',
                                    origin='upper',
                                    cmap=self._ref_color_map,
                                    extent=[int(round(self.crop_x0)),
                                            int(round(self.crop_x1)),
                                            int(round(self.crop_y0)),
                                            int(round(self.crop_y1))])
                tfont = {'size':'22', 'weight':'semibold'}
                self.crop_ax.set_title('ROI will be set as (%d, %d) - (%d, %d)' % (int(round(self.crop_x0)),
                                                                                   int(round(self.crop_y0)),
                                                                                   int(round(self.crop_x1)),
                                                                                   int(round(self.crop_y1))),
                                       **tfont)
                self.crop_canvas.draw()
                self.crop_widget.show()

    def on_motion(self, event):
        if self.set_roi_enabled and event.button==1 and event.inaxes:
            self.rect.set_width(event.xdata - self.crop_x0)
            self.rect.set_height(event.ydata - self.crop_y0)
            self.rect.set_xy((self.crop_x0, self.crop_y0))
            self.ax.figure.canvas.draw()

    def crop_ok(self):
        self.roi_x1_widget.setValue(int(round(self.crop_x0)))
        self.roi_y1_widget.setValue(int(round(self.crop_y0)))
        self.roi_x2_widget.setValue(int(round(self.crop_x1)))
        self.roi_y2_widget.setValue(int(round(self.crop_y1)))
        self.crop_widget.hide()

    def crop_cancel(self):
        self.crop_widget.hide()

    def set_roi_enable(self, pressed):
        if pressed:
            self.rect.set_visible(True)
            self.ax.figure.canvas.draw()
            self.set_roi_enabled = True
        else:
            self.rect.set_visible(False)
            self.ax.figure.canvas.draw()
            self.set_roi_enabled = False

    def update_display(self, a, gx, gy, phi, flag=None): # ax is a pyplot object
        def show_line(ax, line):
            ax.plot(line, '-*')
            #return mpl.pyplot.show()

        #def show_line_T(ax, line):
        #    ax.plot(line)

        def show_image(ax, image):
            #return ax.imshow(np.flipud(image.T), interpolation='nearest',
            #                 origin='upper', cmap=cm.Greys_r)
            return ax.imshow(image, interpolation='nearest',
                             origin='upper', cmap=cm.Greys_r)

        def show_image_line(ax, image, start, end, direction=1):
            if direction == 1:
                ax.axhspan(start, end, facecolor='0.5', alpha=0.5)
                return ax.imshow(image, interpolation='nearest',
                                 origin='upper', cmap=cm.Greys_r)
            if direction == -1:
                ax.axvspan(start, end, facecolor='0.5', alpha=0.5)
                return ax.imshow(image, interpolation='nearest',
                                 origin='upper', cmap=cm.Greys_r)

        tfont = {'size':'28', 'weight':'semibold'}

        plt.hold(True)
        main = DPCWindow.instance
        canvas = self.canvas
        fig = canvas.figure
        fig.clear()
        fig.subplots_adjust(top=0.95, left=0.05, right=0.95, bottom=0.03)

        # Check 2D or 1D mode
        cols_num = main.cols_widget.value()
        rows_num = main.rows_widget.value()
        oned = (cols_num == 1) or (rows_num ==1)

        if oned is True:

            if cols_num is 1:
                gs = gridspec.GridSpec(3, 1)

                canvas.a_ax = a_ax = fig.add_subplot(gs[0, 0])
                a_ax.set_title('Absorption', **tfont)
                canvas.ima = ima = show_line(a_ax, a)

                canvas.gx_ax = gx_ax = fig.add_subplot(gs[1, 0])
                gx_ax.set_title('Phase gradient (x)', **tfont)
                #canvas.imx = imx = show_image(gx_ax, gx)
                canvas.imx = imx = show_line(gx_ax, gx)
                #fig.colorbar(imx)

                canvas.gy_ax = gy_ax = fig.add_subplot(gs[2, 0])
                gy_ax.set_title('Phase gradient (y)', **tfont)
                #canvas.imy = imy = show_image(gy_ax, gy)
                canvas.imy = imy = show_line(gy_ax, gy)
                #fig.colorbar(imy)

            else:
                gs = gridspec.GridSpec(3, 1)

                canvas.a_ax = a_ax = fig.add_subplot(gs[0, 0])
                a_ax.set_title('Absorption', **tfont)
                canvas.ima = ima = show_line(a_ax, a.T)

                canvas.gx_ax = gx_ax = fig.add_subplot(gs[1, 0])
                gx_ax.set_title('Phase gradient (x)', **tfont)
                canvas.imx = imx = show_line(gx_ax, gx.T)

                canvas.gy_ax = gy_ax = fig.add_subplot(gs[2, 0])
                gy_ax.set_title('Phase gradient (y)', **tfont)
                canvas.imy = imy = show_line(gy_ax, gy.T)

        else:
            gs = gridspec.GridSpec(2, 2)

            '''
            if main.ion_data is not None:
                pixels = a.shape[0] * a.shape[1]
                ion_data = np.zeros(pixels)
                ion_data[:len(main.ion_data)] = main.ion_data
                ion_data[len(main.ion_data):] = ion_data[0]
                ion_data = ion_data.reshape(a.shape)

                min_ = np.min(a[np.where(a > 0)])
                a[np.where(a == 0)] = min_

                canvas.a_ax = a_ax = fig.add_subplot(gs[0, 1])
                a_ax.set_title('a')
                a_data = a / ion_data * ion_data[0]
                canvas.ima = ima = show_image(a_ax, a_data)
                fig.colorbar(ima)
            '''

            canvas.a_ax = a_ax = fig.add_subplot(gs[1, 0])
            a_ax.set_title('Absorption', **tfont)
            #a_data = a / ion_data * ion_data[0]
            canvas.ima = ima = show_image(a_ax, a)
            fig.colorbar(ima)
            ima.set_cmap(main._color_map)

            if flag == None:
                canvas.gx_ax = gx_ax = fig.add_subplot(gs[0, 0])
                gx_ax.set_title('Phase gradient (x)', **tfont)
                canvas.imx = imx = show_image(gx_ax, gx)
                #canvas.imx = imx = show_line(gx_ax, gx)
                fig.colorbar(imx)
                imx.set_cmap(main._color_map)

                canvas.gy_ax = gy_ax = fig.add_subplot(gs[0, 1])
                gy_ax.set_title('Phase gradient (y)', **tfont)
                canvas.imy = imy = show_image(gy_ax, gy)
                #canvas.imy = imy = show_line(gy_ax, gy)
                fig.colorbar(imy)
                imy.set_cmap(main._color_map)
            else:
                canvas.gx_ax = gx_ax = fig.add_subplot(gs[0, 0])
                gx_ax.set_title('Phase gradient (x)', **tfont)
                main = DPCWindow.instance
                canvas.imx = imx = show_image_line(gx_ax, gx,
                                                   main.strap_start.value(),
                                                   main.strap_end.value(),
                                                   main.direction)
                fig.colorbar(imx)
                imx.set_cmap(main._color_map)

                canvas.gy_ax = gy_ax = fig.add_subplot(gs[0, 1])
                gy_ax.set_title('Phase gradient (y)', **tfont)
                canvas.imy = imy = show_image_line(gy_ax, gy,
                                                   main.strap_start.value(),
                                                   main.strap_end.value(),
                                                   main.direction)
                fig.colorbar(imy)
                imy.set_cmap(main._color_map)

            if phi is not None:
                phi_ax = fig.add_subplot(gs[1, 1])

                canvas.phi_ax = phi_ax
                phi_ax.set_title('Phase', **tfont)
                canvas.imphi = imphi = show_image(phi_ax, phi)
                fig.colorbar(imphi)
                imphi.set_cmap(main._color_map)

        for splot in fig.axes:
            splot.tick_params(axis='both', which='major', labelsize=21)

        canvas.draw()

    def add_strap(self, pressed):
        """
        Add two lines in the gx and gy
        """
        self.confirm_btn.setEnabled(False)
        self.direction_btn.setEnabled(True)
        self.removal_btn.setEnabled(True)
        self.update_display(a, gx, gy, phi, "strap")

    def change_direction(self, pressed):
        """
        Change the orientation of the strap
        """
        self.direction = -self.direction
        self.update_display(a, gx, gy, phi, "strap")

    def remove_background(self, pressed):
        """
        Remove the background of the phase image
        """
        global gx, gy, phi
        self.confirm_btn.setEnabled(True)
        self.direction_btn.setEnabled(False)
        if self.direction == 1:
            strap_gx = gx[self.strap_start.value():self.strap_end.value(), :]
            line_gx = np.mean(strap_gx, axis=0)
            self.gx_r = gx - line_gx
            strap_gy = gy[self.strap_start.value():self.strap_end.value(), :]
            line_gy = np.mean(strap_gy, axis=0)
            self.gy_r = gy - line_gy
            self.phi_r = dpc.recon(self.gx_r, self.gy_r, self.dx_widget.value(), self.dy_widget.value())
            self.update_display(a, self.gx_r, self.gy_r, self.phi_r)

        if self.direction == -1:
            strap_gx = gx[:, self.strap_start.value():self.strap_end.value()]
            line_gx = np.mean(strap_gx, axis=1)
            self.gx_r = np.transpose(gx)
            self.gx_r = self.gx_r - line_gx
            self.gx_r = np.transpose(self.gx_r)

            strap_gy = gy[:, self.strap_start.value():self.strap_end.value()]
            line_gy = np.mean(strap_gy, axis=1)
            self.gy_r = np.transpose(gy)
            self.gy_r = self.gy_r - line_gy
            self.gy_r = np.transpose(self.gy_r)
            self.phi_r = dpc.recon(self.gx_r, self.gy_r, self.dx_widget.value(), self.dy_widget.value())
            self.update_display(a, self.gx_r, self.gy_r, self.phi_r)

    def confirm(self, pressed):
        """
        Confirm the background removal
        """
        global phi, gx, gy

        phi = self.phi_r
        imsave('phi.jpg', phi)
        np.savetxt('phi.txt', phi)

        gx = self.gx_r
        imsave('gx.jpg', gx)
        np.savetxt('gx.txt', gx)

        gy = self.gy_r
        imsave('gy.jpg', gy)
        np.savetxt('gy.txt', gy)

        self.confirm_btn.setEnabled(False)
        self.direction_btn.setEnabled(False)
        self.removal_btn.setEnabled(False)

    def bad_enable(self, pressed):
        """
        Enable or disable bad pixels selection by changing the bad_flag value
        """
        if pressed:
            self.bad_flag = 1
        else:
            self.bad_flag = 0


    def histgramEqua(self, pressed):
        """
        Histogram equalization for the reference image
        """
        if pressed:
            self.his_enabled = True
            self.ref_canvas.ref_im = self.ax.imshow(self.roi_img_equ, interpolation='nearest', origin='upper', cmap=cm.Greys_r)

        else:
            self.his_enabled = False
            self.ref_canvas.ref_im = self.ax.imshow(self.roi_img, interpolation='nearest', origin='upper', cmap=cm.Greys_r)
        self.ref_canvas.ref_im.set_cmap(self._ref_color_map)
        self.ref_canvas.draw()

    """
    def preContrast(self):
        self.contrastImage = self.roi_img.convert('L')
        #self.contrastImage = self.roi_img
        self.enh = ImageEnhance.Contrast(self.contrastImage)
    """

    def rescale_intensity(self):
        """
        Stretch or shrink ROI image intensity levels
        """
        min = self.min_box.value()
        max = self.max_box.value()
        roi_array = exposure.rescale_intensity(self.roi_img, in_range=(min, max))
        self.ax.imshow(roi_array, interpolation='nearest', origin='upper', cmap=self._ref_color_map)
        self.ref_canvas.draw()

    def calHist(self):
        """
        Calculate the histogram of the image used to select ROI
        """
        imhist,bins = np.histogram(self.roi_img, bins=self.bin_num, range=(0, self.bin_num), density=True)
        cdf = imhist.cumsum()
        cdf = (self.bin_num-1) * cdf / cdf[-1]
        #cdf = (self.roi_img_max-self.roi_img_min) * cdf / cdf[-1]
        equalizedImg = np.floor(np.interp(self.roi_img, bins[:-1], cdf))
        self.roi_img_equ = np.reshape(equalizedImg, self.roi_img.shape, order='C')

        # skimage histgram equalization
        #img = np.array(self.roi_img.getdata(), dtype=np.uint16).reshape(self.roi_img.size[1], self.roi_img.size[0])
        #equalizedImg = exposure.equalize_hist(img)
        #scipy.misc.imsave('equalizedImg.tif', equalizedImg)


    def select_bri_pixels(self):
        """
        Select the bad pixels (pixels with the maximum pixel value)
        """
        indices = np.where(self.roi_img==self.roi_img.max())
        indices_num = indices[0].size
        for i in range(indices_num):
            self.bad_pixels_widget.addItem('%d, %d' % (indices[1][i], indices[0][i]))

    """
    def change_contrast(self, value):
        '''
        Change the contrast of the ROI image by slider bar
        '''
        delta = value / 10.0
        self.enh.enhance(delta).save('change_contrast.tif')
        contrastImageTemp = QtGui.QPixmap('change_contrast.tif')
        self.img_lbl.setPixmap(contrastImageTemp)
    """

    """
    def eventFilter(self, source, event):
        '''
        Event filter to enable cursor coordinates tracking on the ROI image
        '''
        if (event.type() == QtCore.QEvent.MouseMove and
            source is self.ref_canvas):
            if event.buttons() == QtCore.Qt.NoButton:
                pos = event.pos()
                self.txt_lbl.setText('min=%d, max=%d, x=%d, y=%d, value=%d ' %
                                     (self.roi_img_min, self.roi_img_max,
                                      pos.x(), pos.y(),
                                      self.roi_img.getpixel((pos.x(),
                                                             pos.y()))))

                top_left_x = pos.x()-10 if pos.x()-10>=0 else 0
                top_left_y = pos.y()-10 if pos.y()-10>=0 else 0
                bottom_right_x = pos.x()+10 if pos.x()+10<self.roi_img.size[0] else self.roi_img.size[0]-1
                bottom_right_y = pos.y()+10 if pos.y()+10<self.roi_img.size[1] else self.roi_img.size[1]-1

                if (pos.y()-10)<0:
                    self.temp_lbl.setAlignment(QtCore.Qt.AlignBottom)
                if (pos.x()+10)>=self.roi_img.size[0]:
                    self.temp_lbl.setAlignment(QtCore.Qt.AlignLeft)
                if (pos.x()-10)<0:
                    self.temp_lbl.setAlignment(QtCore.Qt.AlignRight)
                if (pos.y()+10)>=self.roi_img.size[1]:
                    self.temp_lbl.setAlignment(QtCore.Qt.AlignTop)

                width = bottom_right_x - top_left_x + 1
                height = bottom_right_y - top_left_y+ 1
                img_fraction = self.img_lbl.pixmap().copy(top_left_x, top_left_y, width, height)
                scaled_img_fraction = img_fraction.scaled(width*8, height*8)
                self.temp_lbl.setPixmap(scaled_img_fraction)

        if (event.type() == QtCore.QEvent.MouseMove and
            source is not self.img_lbl):
            if event.buttons() == QtCore.Qt.NoButton:
                self.txt_lbl.setText('min=%d, max=%d' % (self.roi_img_min,
                                                         self.roi_img_max))
                self.temp_lbl.clear()

        return QtGui.QDialog.eventFilter(self, source, event)
    """

    def select_path(self):
        """
        Select path and initiate file format for the data

        """
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        fname = str(fname)
        if fname != '':
            index1 = fname.rfind('.')
            index2 = fname.rfind('_')
            digits = index1 - index2 - 1
            if digits == 1:
                format_str = '%d'
            else:
                format_str = '%' + '0%dd' % digits
            modified = fname.replace(fname[index2+1:index1], format_str)
            self.file_widget.setText(modified)

    def save_file(self):
        """
        Select the path where the results will be saved

        """
        global a, gx, gy, phi
        path = get_save_filename(self, 'Select path', '/home')
        path = str(path)
        if path != '':
            a_path = path + '_a.txt'
            print(a_path)
            np.savetxt(a_path, a)
            gx_path = path + '_gx.txt'
            np.savetxt(gx_path, gx)
            gy_path = path + '_gy.txt'
            np.savetxt(gy_path, gy)
            if phi is not None:
                phi_path = path + '_phi.txt'
                np.savetxt(phi_path, phi)

    def swap_x_y(self):
        global a, gx, gy, phi
        gx, gy = gy, gx
        phi = dpc.recon(gx, gy, self.dx_widget.value(), self.dy_widget.value())
        self.update_display(a, gx, gy, phi)

    def reverse_gx(self):
        global a, gx, gy, phi
        gx = -gx
        phi = dpc.recon(gx, gy, self.dx_widget.value(), self.dy_widget.value())
        self.update_display(a, gx, gy, phi)

    def reverse_gy(self):
        global a, gx, gy, phi
        gy = -gy
        phi = dpc.recon(gx, gy, self.dx_widget.value(), self.dy_widget.value())
        self.update_display(a, gx, gy, phi)

    def select_ref_img(self):
        """
        Select the reference image and record its location and name

        """
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        fname = str(fname)
        if fname != '':
            self.ref_image_path_QLineEdit.setText(fname)

    def hide_ref(self, pressed):
        """
        Hide/Show the reference image related widgets

        """
        if pressed:
            self.load_img_method()
            if self.first_img_as_ref_checkbox.checkState() == Qt.Unchecked:
                ref_path = str(self.ref_image_path_QLineEdit.text())
            else:
                ref_path = str(self.file_widget.text()) % self.first_widget.value()

            try:
                self.roi_img = self.load_image(ref_path)
                self.calHist()
                self.ref_canvas.ref_im = self.ax.imshow(self.roi_img,
                                                        interpolation='nearest',
                                                        origin='upper',
                                                        cmap=cm.Greys_r)
                self.ref_canvas.ref_im.set_cmap(self._ref_color_map)
                self.ref_widget.show()
                self.ref_canvas.draw()
            except:
                e = sys.exc_info()[1]
                QtGui.QMessageBox.information(self, 'Read error',
                                              '''Could not read the reference image! \r %s''' % (e),
                                              QtGui.QMessageBox.Ok)
                self.hide_btn.setChecked(False)

        else:
            self.ref_widget.hide()

    def first_equal_ref(self, state):
        """
        First image ?= reference image
        """

        if state == QtCore.Qt.Checked:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.lightGray)
            self.ref_image_path_QLineEdit.setPalette(palette)
            self.select_ref_btn.setEnabled(False)
            self.ref_image_path_QLineEdit.setEnabled(False)
        else:
            self.select_ref_btn.setEnabled(True)
            self.ref_image_path_QLineEdit.setEnabled(True)
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
            self.ref_image_path_QLineEdit.setPalette(palette)

    def load_img_method(self):
        method = str(self.img_type_combobox.currentText())

        if method == 'Timepix TIFF':
            self.load_image = load_timepix.load
        elif method == 'TIFF':
            self.load_image = load_image_pil
        elif method == 'ASCII':
            self.load_image = load_image_ascii
        elif method == 'FileStore':
            self.load_image = dpc.load_image_filestore

    def _set_color_map(self, index):
        '''
        User changed color map callback.
        '''
        cm_ = str(self.color_map.itemText(index))
        print('Color map set to: %s' % cm_)
        self._color_map = mpl.cm.get_cmap(cm_)
        try:
            for im in [self.canvas.imphi, self.canvas.imx, self.canvas.ima,
                       self.canvas.imy]:
                im.set_cmap(self._color_map)
        except Exception as ex:
            print('failed to set color map: (%s) %s' % (ex.__class__.__name__,
                                                        ex))
        finally:
            self.canvas.draw()

    def _set_ref_color_map(self, index):
        '''
        User changed color map callback.
        '''
        cm_ = str(self.ref_color_map.itemText(index))
        self._ref_color_map = mpl.cm.get_cmap(cm_)
        try:
            for im in [self.ref_canvas.ref_im, ]:
                im.set_cmap(self._ref_color_map)
        except Exception as ex:
            print('failed to set color map: (%s) %s' % (ex.__class__.__name__,
                                                        ex))
        finally:
            self.ref_canvas.draw()

    def create_cmap_previews(self):
        '''
        Create the color map previews for the combobox
        '''
        cm_names = sorted(_cm for _cm in mpl.cm.datad.keys()
                          if not _cm.endswith('_r'))
        cm_filenames = [os.path.join(CMAP_PREVIEW_PATH, '%s.png' % cm_name)
                        for cm_name in cm_names]

        ret = zip(cm_names, cm_filenames)
        points = np.outer(np.ones(10), np.arange(0, 1, 0.01))
        if not os.path.exists(CMAP_PREVIEW_PATH):
            try:
                os.mkdir(CMAP_PREVIEW_PATH)
            except Exception as ex:
                print('Unable to create preview path: %s' % ex)

            return ret

        for cm_name, fn in zip(cm_names, cm_filenames):
            if not os.path.exists(fn):
                print('Generating colormap preview: %s' % fn)
                canvas = MplCanvas(width=2, height=0.25, dpi=50)
                fig = canvas.figure
                fig.clear()

                ax = fig.add_subplot(1, 1, 1)
                ax.axis("off")
                fig.subplots_adjust(top=1, left=0, right=1, bottom=0)
                _cm = mpl.cm.get_cmap(cm_name)
                ax.imshow(points, aspect='auto', cmap=_cm, origin='upper')
                try:
                    fig.savefig(fn)
                except Exception as ex:
                    print('Unable to create color map preview "%s"' % fn,
                          file=sys.stderr)
                    break

        return ret

    def update_color_maps(self):
        size = None
        for i, (cm_name, fn) in enumerate(self.create_cmap_previews()):
            print('Color map', fn)
            if os.path.exists(fn):
                self.color_map.addItem(QtGui.QIcon(fn), cm_name)
                if size is None:
                    size = QtGui.QPixmap(fn).size()
                    self.color_map.setIconSize(size)
            else:
                self.color_map.addItem(cm_name)

            if cm_name == self.CM_DEFAULT:
                self.color_map.setCurrentIndex(i)

    def update_ref_color_maps(self):
        size = None
        for i, (cm_name, fn) in enumerate(self.create_cmap_previews()):
            if os.path.exists(fn):
                self.ref_color_map.addItem(QtGui.QIcon(fn), cm_name)
                if size is None:
                    size = QtGui.QPixmap(fn).size()
                    self.ref_color_map.setIconSize(size)
            else:
                self.ref_color_map.addItem(cm_name)

            if cm_name == self.CM_DEFAULT:
                self.ref_color_map.setCurrentIndex(i)

    @property
    def settings(self):
        return QtCore.QSettings('BNL', 'DPC-GUI')

    def save_settings(self):
        settings = self.settings
        for key, (getter, setter) in self._settings.items():
            settings.setValue(key, getter())

        settings.setValue('geometry', self.geometry())
        settings.setValue('ref_geo', self.ref_widget.geometry())
        settings.setValue('image_type', self.img_type_combobox.currentIndex())
        settings.setValue('ref_image', str(self.ref_image_path_QLineEdit.text()))
        settings.setValue('first_as_ref', self.first_img_as_ref_checkbox.isChecked())

    def load_settings(self):
        settings = self.settings
        for key, (getter, setter) in self._settings.items():
            try:
                value = settings.value(key).toPyObject()
            except AttributeError:
                continue

            if value is not None:
                setter(value)

        try:
            self.setGeometry(settings.value('geometry').toPyObject())
            self.ref_widget.setGeometry(settings.value('ref_geo').toPyObject())
            self.img_type_combobox.setCurrentIndex(settings.value('image_type').toPyObject())
            self.ref_image_path_QLineEdit.setText(settings.value('ref_image').toPyObject())
            self.first_img_as_ref_checkbox.setChecked(settings.value('first_as_ref').toPyObject())
            self.use_mds.setChecked(settings.value('use_mds').toPyObject())
        except:
            pass

    def closeEvent(self, event=None):
        self.save_settings()
        sys.exit()

    @property
    def dx(self):
        return float(self.dx_widget.text())

    @dx.setter
    def dx(self, dx):
        self.dx_widget.setValue(float(dx))

    @property
    def dy(self):
        return float(self.dy_widget.text())

    @dy.setter
    def dy(self, dy):
        self.dy_widget.setValue(float(dy))

    @property
    def processes(self):
        return int(self.processes_widget.text())

    @property
    def file_format(self):
        return str(self.file_widget.text())

    @property
    def pixel_size(self):
        return self.pixel_widget.value()

    @property
    def focus(self):
        return self.focus_widget.value()

    @property
    def energy(self):
        return self.energy_widget.value()

    @property
    def rows(self):
        return self.rows_widget.value()

    @rows.setter
    def rows(self, rows):
        self.rows_widget.setValue(rows)

    @property
    def cols(self):
        return self.cols_widget.value()

    @cols.setter
    def cols(self, cols):
        self.cols_widget.setValue(cols)

    @property
    def mosaic_x(self):
        return self.mosaic_x_widget.value()

    @property
    def mosaic_y(self):
        return self.mosaic_y_widget.value()

    @property
    def random(self):
        if self.random_processing_opt.isChecked():
            return 1
        else:
            return -1

    @property
    def pyramid(self):
        if self.pyramid_scan.isChecked():
            return 1
        else:
            return -1

    @property
    def swap(self):
        if self.swap_xy.isChecked():
            return 1
        else:
            return -1

    @property
    def re_x(self):
        if self.reverse_x.isChecked():
            return -1
        else:
            return 1

    @property
    def re_y(self):
        if self.reverse_y.isChecked():
            return -1
        else:
            return 1

    @property
    def hang(self):
        if self.hanging_opt.isChecked():
            return 1
        else:
            return -1

    @property
    def first_image(self):
        return self.first_widget.value()

    @property
    def ref_image(self):
        if self.first_img_as_ref_checkbox.checkState() == Qt.Unchecked:
            return str(self.ref_image_path_QLineEdit.text())
        else:
            return str(self.file_widget.text()) % self.first_widget.value()

    @property
    def roi_x1(self):
        return self.roi_x1_widget.value()

    @property
    def roi_x2(self):
        return self.roi_x2_widget.value()

    @property
    def roi_y1(self):
        return self.roi_y1_widget.value()

    @property
    def roi_y2(self):
        return self.roi_y2_widget.value()

    @property
    def bad_pixels(self):
        w = self.bad_pixels_widget

        def fix_tuple(item):
            item = str(item.text())
            return [int(x) for x in item.split(',')]

        return [fix_tuple(w.item(i)) for i in range(w.count())]

    def _bad_pixels_menu(self, pos):
        def add():
            s, ok = QtGui.QInputDialog.getText(self, 'Position?', 'Position in the format: x, y')
            if ok:
                s = str(s)
                x, y = s.split(',')
                x = int(x)
                y = int(y)
                self.bad_pixels_widget.addItem('%d, %d' % (x, y))

        def remove():
            rows = [index.row() for index in
                    self.bad_pixels_widget.selectedIndexes()]
            for row in reversed(sorted(rows)):
                self.bad_pixels_widget.takeItem(row)

        def clear():
            self.bad_pixels_widget.clear()

        self.menu = menu = QtGui.QMenu()
        add_action = menu.addAction('&Add', add)
        remove_action = menu.addAction('&Remove', remove)
        clear_action = menu.addAction('&Clear', clear)

        menu.popup(self.bad_pixels_widget.mapToGlobal(pos))

    def load_from_spec_scan(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Scan filename', self.last_path, '*.spec')
        if not filename:
            return

        self.last_path = filename

        print('Loading %s' % filename)
        with pyspecfile.SPECFileReader(filename, parse_data=False) as f:
            scans = dict((int(scan['number']), scan) for scan in f.scans)
            scan_info = ['%04d - %s' % (number, scan['command'])
                         for number, scan in scans.items()
                         if 'mesh' in scan['command']]

            scan_info.sort()
            print('\n'.join(scan_info))

            s, ok = QtGui.QInputDialog.getItem(self, 'Scan selection', 'Scan number?', scan_info, 0, False)
            if ok:
                print('Selected scan', s)
                number = int(s.split(' ')[0])
                sd = scans[number]
                f.parse_data(sd)

                timepix_index = sd['columns'].index('tpx_image')
                line0 = sd['lines'][0]
                timepix_first_image = int(line0[timepix_index])

                try:
                    ion1_index = sd['columns'].index('Ion1')
                    self.ion_data = np.array([line[ion1_index] for line in sd['lines']])
                except Exception as ex:
                    print('Failed loading Ion1 data (%s) %s' % (ex, ex.__class__.__name__))
                    self.ion_data = None

                print('First timepix image:', timepix_first_image)

                self.first_widget.setValue(timepix_first_image - 1)

                command = sd['command'].replace('  ', ' ')

                x = [2, 3, 4]  # x start, end, points
                y = [6, 7, 8]  # y start, end, points
                info = command.split(' ')

                x_info = [float(info[i]) for i in x]
                y_info = [float(info[i]) for i in y]

                dx = (x_info[1] - x_info[0]) / (x_info[2] - 1)
                dy = (y_info[1] - y_info[0]) / (y_info[2] - 1)

                self.rows_widget.setValue(int(y_info[-1]))
                self.cols_widget.setValue(int(x_info[-1]))

                self.dx_widget.setValue(float(dx))
                self.dy_widget.setValue(float(dy))

    @property
    def solver(self):
        return SOLVERS[self.solver_widget.currentIndex()]

    def set_solver(self, solver):
        self.solver_widget.setCurrentIndex(SOLVERS.index(solver))

    def set_bad_pixels(self, pixels):
        w = self.bad_pixels_widget
        w.clear()
        for item in pixels:
            x, y = item
            w.addItem('%d, %d' % (x, y, ))

    @property
    def dpc_settings(self):
        ret = {}
        for key, (getter, setter) in self._settings.items():
            if key not in ('last_path', 'scan_number', 'filestore_key'):
                ret[key] = getter()
        return ret

    def start(self):
        self.reverse_x.setEnabled(False)
        self.reverse_y.setEnabled(False)
        self.swap_xy.setEnabled(False)
        self.hanging_opt.setEnabled(False)
        self.random_processing_opt.setEnabled(False)
        self.pyramid_scan.setEnabled(False)
        self.save_result.setEnabled(False)
        self.canvas_widget.show()
        self.line_btn.setEnabled(False)
        self.direction_btn.setEnabled(False)
        self.removal_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)

        if self._thread is not None and self._thread.isFinished():
            self._thread = None

        if self._thread is None:
            if self.processes == 0:
                pool = None
            else:
                pool = mp.Pool(processes=self.processes)

            thread = self._thread = DPCThread(self.canvas, pool=pool)
            thread.update_signal.connect(self.update_display)

            thread.dpc_settings = self.dpc_settings
            if self.use_mds:
                thread.dpc_settings['scan'] = self.scan

            # del thread.dpc_settings['processes']
            thread.start()
            self.set_running(True)

    def set_running(self, running):
        self.start_widget.setEnabled(not running)
        self.stop_widget.setEnabled(running)

    def stop(self):
        if self._thread is not None:
            pool = self._thread.pool
            if pool is not None:
                pool.terminate()
                self._thread.pool = None

            time.sleep(0.2)
            self._thread.terminate()
            self._thread = None
            self.set_running(False)

    def save(self):
        filename = get_save_filename(self, 'Save filename prefix', '', '')
        if not filename:
            return

        arrays = [('gx', self.gx),
                  ('gy', self.gy),
                  ('phi', self.phi),
                  ('a', self.a)]

        for name, arr in arrays:
            im = PIL.Image.fromarray(arr)
            im.save('%s_%s.tif' % (filename, name))
            np.savetxt('%s_%s.txt' % (filename, name), im)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    # app.setAttribute(Qt.AA_X11InitThreads)

    window = DPCWindow()
    window.show()
    app.installEventFilter(window)

    sys.exit(app.exec_())
