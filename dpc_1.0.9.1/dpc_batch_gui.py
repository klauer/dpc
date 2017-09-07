'''
Created on Feb 23, 2017

@author: Mirna Lerotic, 2nd Look
'''
from __future__ import division
import sys
import os
import numpy as np

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt, QCoreApplication


import dpc_batch as dpc

version = '1.0.2'

# #----------------------------------------------------------------------
# class EmittingStream(QtCore.QObject):
#
#     textWritten = QtCore.pyqtSignal(str)
#
#     def write(self, text):
#         self.textWritten.emit(str(text))

""" ------------------------------------------------------------------------------------------------"""
class MainFrame(QtGui.QMainWindow):

    def __init__(self):
        super(MainFrame, self).__init__()

        self.settings = QtCore.QSettings('BNL', 'DPC-BATCH-GUI')

        self.script_file = ''

        try:
            val = self.settings.value('scan_range').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = ''
        self.scan_range = val
        try:
            val = self.settings.value('scan_nums').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = ''
        self.scan_nums = val
        try:
            val = self.settings.value('every_n').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = 1
        self.every_n = val
        try:
            val = self.settings.value('load_params_datastore').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = 0
        self.read_data_from_datastore = val
        try:
            val = self.settings.value('filestore_key').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = 'merlin1'
        self.filestore_key = val
        try:
            val = self.settings.value('data_dir').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = ''
        self.data_directory = val
        try:
            val = self.settings.value('file_format').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = 'S{0}.h5'
        self.file_format = val
        try:
            val = self.settings.value('load_params_datastore').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = 0
        self.load_params_from_broker = val
        try:
            val = self.settings.value('param_file').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = ''
        self.parameter_file = val
        try:
            val = self.settings.value('processes').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = 1
        self.processes = val
        try:
            val = self.settings.value('save_dir').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = ''
        self.save_dir = val
        try:
            val = self.settings.value('save_fn').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = ''
        self.save_filename = val
        try:
            val = self.settings.value('save_png').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = 0
        self.save_png = val
        try:
            val = self.settings.value('save_txt').toPyObject()
        except AttributeError:
            val = None
        if val is None:
            val = 0
        self.save_txt = val


        self.resize(600, 720)
        self.setWindowTitle('DPC Batch v.{}'.format(version))

        pal = QtGui.QPalette()
        self.setAutoFillBackground(True)
        pal.setColor(QtGui.QPalette.Window,QtGui.QColor('white'))
        self.setPalette(pal)

        self.mainWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.mainWidget)

        vbox = QtGui.QVBoxLayout(self.mainWidget)
        vbox.setContentsMargins(20, 10, 20, 20)

        sizer1 = QtGui.QGroupBox('Scans')
        vbox1 = QtGui.QVBoxLayout()

        self.cb_usedatastore = QtGui.QCheckBox( '  Read the Data from DataStore', self)
        self.cb_usedatastore.setChecked(self.read_data_from_datastore)
        self.cb_usedatastore.stateChanged.connect(self.OnUseDataStore)
        vbox1.addWidget(self.cb_usedatastore)

        hbox = QtGui.QHBoxLayout()
        l1 = QtGui.QLabel('Scan numbers & ranges \t', self)
        self.tc_scan_range = QtGui.QLineEdit(self)
        self.tc_scan_range.setAlignment(QtCore.Qt.AlignLeft)
        self.tc_scan_range.setText(self.scan_range)
        l1.setToolTip('Set scan numbers and ranges. Example: 2, 3-5, 7-15, 23, 30-55')
        self.tc_scan_range.setToolTip('Set scan numbers and ranges. Example: 2, 3-5, 7-15, 23, 30-55')
        hbox.addWidget(l1)
        hbox.addWidget(self.tc_scan_range)
        vbox1.addLayout(hbox)

        hbox = QtGui.QHBoxLayout()
        l2 = QtGui.QLabel('Process every n-th scan \t', self)
        self.ntc_every_n = QtGui.QLineEdit(self)
        self.ntc_every_n.setValidator(QtGui.QIntValidator(1, 99999, self))
        self.ntc_every_n.setAlignment(QtCore.Qt.AlignRight)
        self.ntc_every_n.setText(str(self.every_n))
        hbox.addWidget(l2)
        hbox.addWidget(self.ntc_every_n)
        hbox.addStretch(1)
        vbox1.addLayout(hbox)

        # hbox = QtGui.QHBoxLayout()
        # l1 = QtGui.QLabel('Scan numbers \t', self)
        # self.tc_scans = QtGui.QLineEdit(self)
        # self.tc_scans.setAlignment(QtCore.Qt.AlignLeft)
        # self.tc_scans.setText(str(self.scan_nums))
        # l1.setToolTip('Set scan numbers. Example: 1, 24, 26')
        # self.tc_scans.setToolTip('Set scan numbers. Example: 1, 24, 26')
        # hbox.addWidget(l1)
        # hbox.addWidget(self.tc_scans)
        # vbox1.addLayout(hbox)

        hbox = QtGui.QHBoxLayout()
        l1 = QtGui.QLabel('Filestore key \t', self)
        self.tc_fskey = QtGui.QLineEdit(self)
        self.tc_fskey.setAlignment(QtCore.Qt.AlignLeft)
        self.tc_fskey.setText(self.filestore_key)
        hbox.addWidget(l1)
        hbox.addWidget(self.tc_fskey)
        vbox1.addLayout(hbox)

        hbox = QtGui.QHBoxLayout()
        l1 = QtGui.QLabel('Data Directory \t', self)
        self.tc_datadir = QtGui.QLineEdit(self)
        self.tc_datadir.setAlignment(QtCore.Qt.AlignLeft)
        l1.setToolTip('Data Direcory if not using datastore.')
        self.tc_datadir.setToolTip('Data Direcory if not using datastore.')
        self.tc_datadir.setText(self.data_directory)
        self.button_d1 = QtGui.QPushButton('Browse')
        self.button_d1.clicked.connect(self.OnSelectDataDir)
        hbox.addWidget(l1)
        hbox.addWidget(self.tc_datadir)
        hbox.addWidget(self.button_d1)
        vbox1.addLayout(hbox)

        hbox = QtGui.QHBoxLayout()
        l1 = QtGui.QLabel('File format \t', self)
        self.tc_format = QtGui.QLineEdit(self)
        self.tc_format.setAlignment(QtCore.Qt.AlignLeft)
        self.tc_format.setToolTip('Data file format.')
        self.tc_format.setText(self.file_format)
        hbox.addWidget(l1)
        hbox.addWidget(self.tc_format)
        vbox1.addLayout(hbox)

        sizer1.setLayout(vbox1)
        vbox.addWidget(sizer1)

        sizer3 = QtGui.QGroupBox('Scan Parameters')
        vbox3 = QtGui.QVBoxLayout()

        self.cb_paramsdatastore = QtGui.QCheckBox( '  Read the Parameters from DataStore', self)
        self.cb_paramsdatastore.setChecked(self.load_params_from_broker)
        vbox3.addWidget(self.cb_paramsdatastore)

        hbox = QtGui.QHBoxLayout()
        l1 = QtGui.QLabel('Parameter file \t', self)
        self.tc_paramfile = QtGui.QLineEdit(self)
        self.tc_paramfile.setAlignment(QtCore.Qt.AlignLeft)
        self.tc_paramfile.setText(self.parameter_file)
        button_d3 = QtGui.QPushButton('Select')
        button_d3.clicked.connect(self.OnSelectParamFile)
        hbox.addWidget(l1)
        hbox.addWidget(self.tc_paramfile)
        hbox.addWidget(button_d3)
        vbox3.addLayout(hbox)

        sizer3.setLayout(vbox3)
        vbox.addWidget(sizer3)

        sizer2 = QtGui.QGroupBox('Run configurations')
        vbox2 = QtGui.QVBoxLayout()
        hbox = QtGui.QHBoxLayout()

        l2 = QtGui.QLabel('Processes  \t', self)
        self.ntc_processes = QtGui.QLineEdit(self)
        self.ntc_processes.setValidator(QtGui.QIntValidator(1, 64, self))
        self.ntc_processes.setAlignment(QtCore.Qt.AlignRight)
        self.ntc_processes.setText(str(self.processes))
        hbox.addWidget(l2)
        hbox.addWidget(self.ntc_processes)
        hbox.addStretch(1)
        vbox2.addLayout(hbox)

        sizer2.setLayout(vbox2)
        vbox.addWidget(sizer2)

        sizer4 = QtGui.QGroupBox('Saving Results')
        vbox4 = QtGui.QVBoxLayout()
        hbox = QtGui.QHBoxLayout()

        hbox = QtGui.QHBoxLayout()
        l1 = QtGui.QLabel('Save Directory \t', self)
        self.tc_savedir = QtGui.QLineEdit(self)
        self.tc_savedir.setAlignment(QtCore.Qt.AlignLeft)
        l1.setToolTip('Data Direcory where results will be stored.')
        self.tc_savedir.setToolTip('Data Direcory where results will be stored.')
        self.tc_savedir.setText(self.save_dir)
        button_d2 = QtGui.QPushButton('Browse')
        button_d2.clicked.connect(self.OnSelectSaveDir)
        hbox.addWidget(l1)
        hbox.addWidget(self.tc_savedir)
        hbox.addWidget(button_d2)
        vbox4.addLayout(hbox)

        hbox = QtGui.QHBoxLayout()
        l1 = QtGui.QLabel('Save Filename \t', self)
        self.tc_savefn = QtGui.QLineEdit(self)
        self.tc_savefn.setAlignment(QtCore.Qt.AlignLeft)
        self.tc_savefn.setText(self.save_filename)
        hbox.addWidget(l1)
        hbox.addWidget(self.tc_savefn)
        vbox4.addLayout(hbox)

        self.cb_savepng = QtGui.QCheckBox('  Save results as .png files', self)
        self.cb_savepng.setChecked(self.save_png)
        vbox4.addWidget(self.cb_savepng)

        self.cb_savetxt = QtGui.QCheckBox('  Save results as .txt files', self)
        self.cb_savetxt.setChecked(self.save_txt)
        vbox4.addWidget(self.cb_savetxt)

        cb_savetif = QtGui.QCheckBox('  Save results as .tif files', self)
        cb_savetif.setChecked(True)
        cb_savetif.setDisabled(True)
        vbox4.addWidget(cb_savetif)

        sizer4.setLayout(vbox4)
        vbox.addWidget(sizer4)

        hbox = QtGui.QHBoxLayout()
        self.button_save = QtGui.QPushButton( 'Save')
        self.button_save.clicked.connect( self.OnSave)
        hbox.addWidget(self.button_save)
        self.button_start = QtGui.QPushButton( 'Start')
        self.button_start.clicked.connect( self.OnStart)
        hbox.addWidget(self.button_start)
        vbox.addLayout(hbox)

        self.console_info = QtGui.QTextEdit(self)
        self.console_info.setReadOnly(True)
        vbox.addWidget(self.console_info)

        #sys.stdout = EmittingStream(textWritten=self.ConsoleOutput)

        self.show()
        if sys.platform == "darwin":
            self.raise_()

        self.OnUseDataStore()


    # ----------------------------------------------------------------------
    def __del__(self):
        sys.stdout = sys.__stdout__


    # ----------------------------------------------------------------------
    def ConsoleOutput(self, text):

        cursor = self.console_info.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.console_info.setTextCursor(cursor)
        self.console_info.ensureCursorVisible()

#----------------------------------------------------------------------
    def OnBrowseDir(self):

        directory = QtGui.QFileDialog.getExistingDirectory(self, "Choose a directory", '', QtGui.QFileDialog.ShowDirsOnly|QtGui.QFileDialog.ReadOnly)

        if directory == '':
            return ''

        return str(directory)

#----------------------------------------------------------------------
    def OnSelectDataDir(self):
        datapath = self.OnBrowseDir()
        self.tc_datadir.setText(str(os.path.abspath(datapath)))


#----------------------------------------------------------------------
    def OnSelectSaveDir(self):
        datapath = self.OnBrowseDir()
        self.tc_savedir.setText(str(os.path.abspath(datapath)))


#----------------------------------------------------------------------
    def OnSelectParamFile(self):
        paramfile = QtGui.QFileDialog.getOpenFileName(self, "Choose a parameter file", '', "Text file (*.txt)")
        self.tc_paramfile.setText(str(os.path.abspath(paramfile)))


#----------------------------------------------------------------------
    def OnUseDataStore(self):
        if self.cb_usedatastore.isChecked():
            self.read_data_from_datastore = 1
            self.tc_datadir.setDisabled(True)
            self.button_d1.setDisabled(True)
            self.tc_format.setDisabled(True)
            self.ntc_every_n.setDisabled(False)
            self.tc_fskey.setDisabled(False)
        else:
            self.read_data_from_datastore = 0
            self.tc_datadir.setDisabled(False)
            self.button_d1.setDisabled(False)
            self.tc_format.setDisabled(False)
            self.ntc_every_n.setDisabled(True)
            self.tc_fskey.setDisabled(True)


#----------------------------------------------------------------------
    def OnStart(self, evt):

        self.console_info.append('Started DPC batch...')
        QCoreApplication.processEvents()

        QtGui.QApplication.setOverrideCursor(Qt.WaitCursor)
        self.script_file = 'DPCBatchGUIScriptFile.txt'
        self.Save(self.script_file)
        dpc.run_batch(self.script_file)

        QtGui.QApplication.restoreOverrideCursor()
        self.console_info.append('DPC finished.')


#----------------------------------------------------------------------
    def OnSave(self, evt):

        scriptfile = QtGui.QFileDialog.getSaveFileName(self, "Choose a script file", '', "Text file (*.txt)")

        if scriptfile == '':
            return

        scriptfile = os.path.abspath(scriptfile)

        self.Save(scriptfile)


#----------------------------------------------------------------------
    def Save(self, scriptfile):

        #Get the info
        self.scan_range = self.tc_scan_range.text()
        self.settings.setValue('scan_range', self.scan_range)
        # self.scan_nums = self.tc_scans.text()
        # self.settings.setValue('scan_nums', self.scan_nums)
        if (self.scan_range == ''):
            QtGui.QMessageBox.warning(self, "Error", "Please enter scan range or scan number.")
            return

        self.every_n = self.ntc_every_n.text()
        self.settings.setValue('every_n', self.every_n)
        if self.cb_usedatastore.isChecked():
            self.read_data_from_datastore = 1
        else:
            self.read_data_from_datastore = 0
        self.settings.setValue('read_data_from_datastore', self.read_data_from_datastore)
        self.filestore_key = self.tc_fskey.text()
        self.settings.setValue('filestore_key', self.filestore_key)
        self.data_directory = self.tc_datadir.text()
        self.settings.setValue('data_dir', self.data_directory)
        if (self.data_directory == '') and (self.read_data_from_datastore == 0):
            QtGui.QMessageBox.warning(self, "Error", "Please enter data directory or read from DataStore.")
            return
        self.file_format = self.tc_format.text()
        self.settings.setValue('file_format', self.file_format)
        if self.cb_paramsdatastore.isChecked():
            self.load_params_from_broker = 1
        else:
            self.load_params_from_broker = 0
        self.settings.setValue('load_params_datastore', self.load_params_from_broker)
        self.parameter_file = self.tc_paramfile.text()
        self.settings.setValue('param_file', self.parameter_file)
        if self.parameter_file == '':
            QtGui.QMessageBox.warning(self, "Error", "Please enter scan parameter file.")
            return
        self.processes = self.ntc_processes.text()
        self.settings.setValue('processes', self.processes)
        self.save_dir = self.tc_savedir.text()
        if self.save_dir == '':
            QtGui.QMessageBox.warning(self, "Error", "Please enter save directory.")
            return
        self.settings.setValue('save_dir', self.save_dir)
        self.save_filename = self.tc_savefn.text()
        if self.save_filename == '':
            QtGui.QMessageBox.warning(self, "Error", "Please enter save file name.")
            return
        self.settings.setValue('save_fn', self.save_filename)
        if self.cb_savepng.isChecked():
            self.save_png = 1
        else:
            self.save_png = 0
        self.settings.setValue('save_png', self.save_png)
        if self.cb_savetxt.isChecked():
            self.save_txt = 1
        else:
            self.save_txt = 0
        self.settings.setValue('save_txt', self.save_txt)


        #Save the info into script file
        self.console_info.append('\n#DPC script file')
        if self.scan_range != '':
            self.console_info.append( 'scan_range = {0}'.format(self.scan_range))
        # if self.scan_nums != '':
        #     self.console_info.append('scan_numbers = {0}'.format(self.scan_nums))
        self.console_info.append( 'every_nth_scan = {0}'.format(self.every_n))
        self.console_info.append( 'get_data_from_datastore = {0}'.format(self.read_data_from_datastore))
        self.console_info.append( 'file_store_key = {0}'.format(self.filestore_key))
        self.console_info.append('data_directory = {0}'.format(self.data_directory))
        self.console_info.append( 'file_format = {0}'.format(self.file_format))
        self.console_info.append( 'parameter_file = {0}'.format(self.parameter_file))
        self.console_info.append( 'read_params_from_datastore = {0}'.format(self.load_params_from_broker))
        self.console_info.append( 'processes = {0}'.format(self.processes))
        self.console_info.append( 'save_path = {0}'.format(self.save_dir))
        self.console_info.append( 'save_filename = {0}'.format(self.save_filename))
        self.console_info.append( 'save_pngs = {0}'.format(self.save_png))
        self.console_info.append( 'save_txt = {0}'.format(self.save_txt))

        try:
            sf = open(scriptfile, "w")
            sf.write('#DPC script file\n')
            if self.scan_range != '':
                sf.write('scan_range = {0}\n'.format(self.scan_range))
            # if self.scan_nums != '':
            #     sf.write('scan_numbers = {0}\n'.format(self.scan_nums))
            sf.write('every_nth_scan = {0}\n'.format(self.every_n))
            sf.write('get_data_from_datastore = {0}\n'.format(self.read_data_from_datastore))
            sf.write('file_store_key = {0}\n'.format(self.filestore_key))
            sf.write('data_directory = {0}\n'.format(self.data_directory))
            sf.write('file_format = {0}\n'.format(self.file_format))
            sf.write('parameter_file = {0}\n'.format(self.parameter_file))
            sf.write('read_params_from_datastore = {0}\n'.format(self.load_params_from_broker))
            sf.write('processes = {0}\n'.format(self.processes))
            sf.write('save_path = {0}\n'.format(self.save_dir))
            sf.write('save_filename = {0}\n'.format(self.save_filename))
            sf.write('save_pngs = {0}\n'.format(self.save_png))
            sf.write('save_txt = {0}\n'.format(self.save_txt))
            sf.close()

            self.console_info.append('\nSaved script file {0}'.format(scriptfile))
        except:
            QtGui.QMessageBox.warning(self, "Error",'Error writing script file!')
            return

        self.script_file = scriptfile


""" ------------------------------------------------------------------------------------------------"""
def main():

    app = QtGui.QApplication(sys.argv)
    frame = MainFrame()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
