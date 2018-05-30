#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui

import logging
import json
import argparse
import sys
import os
import PIL.Image
import tensorflow as tf
import numpy as np

from main_window import Ui_main_window
import config

logger = logging.getLogger (__name__)

class DNNPosition:
    def __init__ (self):
        self.estimator = tf.estimator.Estimator(
                        model_fn=self.ModelFN,
                        params={
                            "layers": [4096, 2048, 512, 256, 256, 4],
                            #"layers": [2048, 512, 256, 128, 64, 32, 4],
                            "learning_rate": 0.001,
                            "optimizer": tf.train.AdamOptimizer
                        },
                        model_dir="/home/harald/NoBackup/Würfel/ModelDir")

    @staticmethod
    def ModelFN (features, labels, mode, params):
      """Model function."""
      # Input Layer
      layers = params.get ("layers", [32, 4])
    
      layer = tf.reshape (features["images_small"], [-1, 92 * 69])
    
      for lsize in layers:
          layer = tf.layers.dense (inputs=layer, units=lsize, activation=tf.nn.relu)
    
      predictions = tf.squeeze (layer)
    
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec (mode=mode, predictions={"position": predictions})
    
      # Calculate Loss (for both TRAIN and EVAL modes)
      average_loss = tf.losses.mean_squared_error (labels, predictions)
    
      # Pre-made estimators use the total_loss instead of the average,
      # so report total_loss for compatibility.
      batch_size = tf.shape (labels)[0]
      total_loss = tf.to_float (batch_size) * average_loss
    
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get ("optimizer", tf.train.AdamOptimizer)
        optimizer = optimizer (params.get ("learning_rate", None))
        train_op = optimizer.minimize (loss=average_loss, global_step=tf.train.get_global_step ())
    
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
    
      # In evaluation mode we will calculate evaluation metrics.
      assert mode == tf.estimator.ModeKeys.EVAL
    
      # Calculate root mean squared error
      rmse = tf.metrics.root_mean_squared_error (labels, predictions)
    
      # Add the rmse to the collection of evaluation metrics.
      eval_metrics = {"rmse": rmse}
    
      return tf.estimator.EstimatorSpec (
          mode=mode,
          # Report sum of error for compatibility with pre-made estimators
          loss=total_loss,
          eval_metric_ops=eval_metrics)
    
    def train (self, train_input_function, eval_input_function, hooks=None, iterations=1):
        r = { 'global_step': 0,
              'loss': 0.0,
              'rmse': 0.0 }
        
        for i in range (iterations):
            self.estimator.train (input_fn=train_input_function, hooks=hooks, steps=30)
            r = self.estimator.evaluate (input_fn=eval_input_function, hooks=hooks, steps=1)
            print ("steps: {} loss: {} rmse: {}".format (r['global_step'], r['loss'], r['rmse']))

        return r

    def eval (self, input_function, hooks=None):
        pos_x1, pos_y1, pos_x2, pos_y2 = [ v['position'] for v in self.estimator.predict (input_fn=input_function, predict_keys="position", hooks=hooks) ]

        return (pos_x1, pos_y1, pos_x2, pos_y2)
        #return self.estimator.evaluate (input_fn=input_function, steps=1)
        
class TrainThread (QtCore.QThread):
    iterationSignal = QtCore.pyqtSignal (int, float)
    finishedSignal = QtCore.pyqtSignal ()
    
    def __init__ (self, trainer):
        QtCore.QThread.__init__ (self)
        self.trainer = trainer
        self.abort = False
    
    def run (self):
        while not self.abort:
            r = self.trainer ()
            self.iterationSignal.emit (r['global_step'], r['rmse'])

        self.finishedSignal.emit ()
         
    def abortTraining (self):
        self.abort = True

class SnapThread (QtCore.QThread):
    imageSnappedSignal = QtCore.pyqtSignal ("QString")

    def __init__ (self, fname):
        QtCore.QThread.__init__ (self)
        self.fname = fname
    
    def run (self):
        try:
            cmd = config.cfg.capture_command
            p = QtCore.QProcess ()
            #print ("start", repr (cmd))
            p.start (cmd)

            p.waitForStarted(-1)
            
            f = open (self.fname, "wb")
            while p.state () != QtCore.QProcess.NotRunning:
                if p.waitForReadyRead (-1):
                    b = p.read (1024*1024)
                
                    if len (b) > 0:
                        f.write (b)
    
            f.close ()
            self.imageSnappedSignal.emit (self.fname)
        except Exception as e:
            self.imageSnappedSignal.emit ("")
           
class ImageEntry:
    def __init__ (self, file_name, value = None, pos_x = None, pos_y = None, pos_x2 = None, pos_y2 = None):
        self.file_name = file_name
        if pos_x2 == None and pos_x != None:
            self.pos_x1 = pos_x - 225
            self.pos_x2 = pos_x + 225
        else:
            self.pos_x1 = pos_x
            self.pos_x2 = pos_x2
            
        if pos_y2 == None and pos_y != None:
            self.pos_y1 = pos_y - 225
            self.pos_y2 = pos_y + 225
        else:
            self.pos_y1 = pos_y
            self.pos_y2 = pos_y2
            
        self.value = value
        self.cached_data = {}
        
    def hasPos (self):
        return self.pos_x1 != None and self.pos_y1 != None and self.pos_x2 != None and self.pos_y2 != None
    
    def hasValue (self):
        return self.value != None
    
    def hasCachedData (self, name):
        return name in self.cached_data
    
    def getCachedData (self, name):
        return self.cached_data[name]
    
    def setCachedData (self, name, data):
        self.cached_data[name] = data

class ImagesModel (QtGui.QStandardItemModel):
    
    IMAGE_NAME, POS_X1, POS_Y1, POS_X2, POS_Y2, VALUE, COLUMNS = range (7)
    
    def __init__ (self, main_window):
        QtGui.QStandardItemModel.__init__ (self, 0, 6)
        self.setDataList([], "")
        self.main_window = main_window
        
    def setDataList (self, data, base_dir):
        self.base_dir = base_dir
        self.beginResetModel ()
        self.data_list = data
        self.endResetModel ()
        self.setRowCount (len (data))
        self.dataChanged.emit (self.index (0,0), self.index (len (data)-1, ImagesModel.COLUMNS-1))
        
    def addNewImage (self, image_entry):
        pos = len (self.data_list)
        self.beginInsertRows (QtCore.QModelIndex (), pos, pos)
        self.data_list.append (image_entry)
        self.endInsertRows ()
        self.setRowCount (len (self.data_list))
        self.dataChanged.emit (self.index (pos,0), self.index (pos, ImagesModel.COLUMNS-1))
        return pos

    def loadSmallImage (self, fname, *label):
        #print ("###### fname:", fname)
        label = [ tf.cast (l, dtype=tf.float32) for l in label ]

        image_string = tf.read_file (fname)
        image = tf.image.decode_jpeg (image_string)
        image = tf.image.convert_image_dtype (image, dtype=tf.float32)
        image = tf.image.resize_images (image, [92, 69])
        image = tf.image.rgb_to_grayscale (image)
        image = tf.reshape (image, (6348, ))
        return ( {"images_small": image }, *label )
    
    def getPosData (self, rows, batches, repeat):
        filenames = [ os.path.join (self.base_dir, self.data_list[row].file_name) for row in rows ]
        numbers = [ self.data_list[row].value for row in rows ]
        pos = [ (self.data_list[row].pos_x1, self.data_list[row].pos_y1,
                 self.data_list[row].pos_x2, self.data_list[row].pos_y2) for row in rows ]
    
        dataset = tf.data.Dataset.from_tensor_slices ( (filenames, pos) )
        dataset = dataset.map (self.loadSmallImage)
        dataset = dataset.shuffle (buffer_size=1)
        dataset = dataset.batch (batches)
        dataset = dataset.repeat (repeat)
        
        iterator = dataset.make_one_shot_iterator()
    
        # `features` is a dictionary in which each value is a batch of values for
        # that feature; `labels` is a batch of labels.
        features, labels = iterator.get_next()
        return features, labels

    def getTrainingPosData (self):
        return self.getPosData (range (len (self.data_list)), 11, 10)
    
    def getEvalPosData (self):
        return self.getPosData (range (len (self.data_list)), 22, 1)
    
    def getPredictPosData (self, row):
        return self.getPosData ([ row ], 1, 1)
    
    def columnCount (self, parent_index):
        if parent_index == QtCore.QModelIndex ():
            return ImagesModel.COLUMNS
        
        return 0
    
    def headerData (self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal:
            if role == QtCore.Qt.DisplayRole:
                if section == ImagesModel.IMAGE_NAME:
                    return "Image"
                
                elif section == ImagesModel.POS_X1:
                    return "X1"
                
                elif section == ImagesModel.POS_Y1:
                    return "Y1"
                
                elif section == ImagesModel.POS_X2:
                    return "X2"
                
                elif section == ImagesModel.POS_Y2:
                    return "Y2"
                
                elif section == ImagesModel.VALUE:
                    return "Value"
                
        return None

    def data (self, index, role):
        if index.row () < len (self.data_list):
            row = index.row ()
            col = index.column ()
            if role == QtCore.Qt.DisplayRole:
                if col == ImagesModel.IMAGE_NAME:
                    return os.path.basename (self.data_list[row].file_name)
                
                elif col == ImagesModel.POS_X1:
                    if self.data_list[row].hasPos ():
                        return "{}".format (self.data_list[row].pos_x1)
                    else:
                        return "---"
                    
                elif col == ImagesModel.POS_Y1:
                    if self.data_list[row].hasPos ():
                        return "{}".format (self.data_list[row].pos_y1)
                    else:
                        return "---"
                    
                elif col == ImagesModel.POS_X2:
                    if self.data_list[row].hasPos ():
                        return "{}".format (self.data_list[row].pos_x2)
                    else:
                        return "---"
                    
                elif col == ImagesModel.POS_Y2:
                    if self.data_list[row].hasPos ():
                        return "{}".format (self.data_list[row].pos_y2)
                    else:
                        return "---"
                    
                elif col == ImagesModel.VALUE:
                    if self.data_list[row].hasValue ():
                        return "{}".format (self.data_list[row].value)
                    else:
                        return "---"
                
            elif role == QtCore.Qt.TextAlignmentRole:
                if col == ImagesModel.IMAGE_NAME:
                    return QtCore.Qt.AlignLeft
                else:
                    return QtCore.Qt.AlignRight
        
        return None
    
class ImageWidget (QtWidgets.QGraphicsView):
    MAX_SCALING = 30.0
    
    rectSelected = QtCore.pyqtSignal (QtCore.QPointF, QtCore.QPointF)

    def __init__ (self, parent):
        QtWidgets.QGraphicsView.__init__ (self, parent)
        self.pixmap = None
        self.min_scaling = 1.0
        self.dice_position = None
        self.reset_scaling_on_next_paint = False
        self.start_rect_selection = None

    def setImage (self, pixmap):
        self.reset_scaling_on_next_paint = self.pixmap == None

        self.pixmap = pixmap

        if self.pixmap == None:
            self.min_scaling = 1.0

        fx = self.width () / self.pixmap.width ()
        fy = self.height () / self.pixmap.height ()
        self.min_scaling = min ((fx, fy))
        
        self.changeScene ()
            
    def setDicePosition (self, pos):
        self.dice_position = pos
        self.changeScene ()
        #print ("Dice position:", pos)
        
    def changeScene (self):
        scene = QtWidgets.QGraphicsScene (self)
        if self.pixmap == None:
            self.setScene (scene)
            self.scale (1.0, 1.0)
            return
        
        scene.addPixmap (self.pixmap)
        width = self.pixmap.width ()
        height = self.pixmap.height ()
        
        if self.dice_position != None:
            pen = QtGui.QPen ()
            pen.setWidth (3)
            pen.setColor (QtGui.QColor (100, 200, 100))
            scene.addLine (self.dice_position[0], self.dice_position[1], self.dice_position[0], self.dice_position[3], pen)
            scene.addLine (self.dice_position[2], self.dice_position[1], self.dice_position[2], self.dice_position[3], pen)
            scene.addLine (self.dice_position[0], self.dice_position[1], self.dice_position[2], self.dice_position[1], pen)
            scene.addLine (self.dice_position[0], self.dice_position[3], self.dice_position[2], self.dice_position[3], pen)
            
        self.setScene (scene)
        
        if self.reset_scaling_on_next_paint:
            self.reset_scaling_on_next_paint = False
            self.setScaling (self.min_scaling)
        else:
            self.changeScaling (1.0)

    def mousePressEvent (self, event):
        if event.modifiers () == QtCore.Qt.ControlModifier and self.pixmap != None:
            #print ("Mouse click @ {}, {}".format (event.x (), event.y ()))
            p = self.mapToScene (event.x (), event.y ())
            #print ("Mouse click @ {}, {}".format (p.x (), p.y ()))
            self.start_rect_selection = p
            return
        
        self.start_rect_selection = None
        return QtWidgets.QGraphicsView.mousePressEvent (self, event)
            
    def mouseReleaseEvent (self, event):
        if self.start_rect_selection != None and event.modifiers () == QtCore.Qt.ControlModifier and self.pixmap != None:
            start_rect_selection = self.start_rect_selection
            self.start_rect_selection = None
            
            #print ("Mouse click @ {}, {}".format (event.x (), event.y ()))
            p = self.mapToScene (event.x (), event.y ())
            #print ("Mouse click @ {}, {}".format (p.x (), p.y ()))

            self.rectSelected.emit (start_rect_selection, p)
            return
        
        return QtWidgets.QGraphicsView.mouseReleaseEvent (self, event)
        
    def wheelEvent (self, event):
        if event.modifiers () == QtCore.Qt.ControlModifier:
            up_down = event.angleDelta ().y ()
            left_right = event.angleDelta ().x ()
             
            if up_down != 0:
                if up_down > 0:
                    self.changeScaling (1.1, QtCore.QPoint (event.x (), event.y ()))

                elif up_down < 0:
                    self.changeScaling (1 / 1.1, QtCore.QPoint (event.x (), event.y ()))
                
                return
            
        return QtWidgets.QGraphicsView.wheelEvent (self, event)
                
    def changeScaling (self, rel_factor, keep_point = None):
        t_old = self.transform ()
        old_sf = 1.0 / t_old.m33 ()
        
        abs_sf = old_sf * rel_factor
        
        if abs_sf < self.min_scaling:
            abs_sf = self.min_scaling
            rel_factor = abs_sf / old_sf
        elif abs_sf > ImageWidget.MAX_SCALING:
            abs_sf = ImageWidget.MAX_SCALING
            rel_factor = abs_sf / old_sf
        
        t = QtGui.QTransform (1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0/rel_factor)
        
        #print ("rel {} * old {} = {}".format (rel_factor, old_sf, rel_factor * old_sf))
        self.setTransform (t, True)

        # Folgendes funktioniert leider noch nicht. Es soll eigentlich die angegebene Position erhalten bleiben
        # (Mausposition beim Zoom mit dem Mausrad)
        if keep_point != None:
            dx = keep_point.x () * (abs_sf - 1) - t_old.m31 ()
            dy = keep_point.y () * (abs_sf - 1) - t_old.m32 ()
            #print ("d: {}, {} (old: {}, {})".format (dx, dy, t_old.m31 (), t_old.m32 ()))

            #self.scrollContentsBy (int (dx), int (dy))
            
    def setScaling (self, abs_factor):
        if abs_factor < self.min_scaling:
            abs_factor = self.min_scaling
        elif abs_factor > ImageWidget.MAX_SCALING:
            abs_factor = ImageWidget.MAX_SCALING

        t = QtGui.QTransform (1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0 / abs_factor)
        
        self.setTransform (t, False)

class SessionRunHook (tf.train.SessionRunHook):
    def __init__ (self, main_window):
        self.main_window = main_window
        
    def after_create_session (self, session, coord):
        #print ("create session:", repr (session))
        self.main_window.setTFSession (session)
        
    def end (self, session):
        #print ("end session:", repr (session))
        self.main_window.setTFSession (None)
        
class MainWindow (QtWidgets.QMainWindow):
    ui = Ui_main_window ()
    
    def __init__ (self):
        self.image_base_dir = ""
        self.img = None
        QtWidgets.QMainWindow.__init__ (self)
        self.ui = Ui_main_window ()
        self.ui.setupUi (self)
        self.ui.dice_value_sb.setMinimum (0)
        self.ui.dice_value_sb.setMaximum (20)
        self.ui.snap_image_btn.clicked.connect (self.snapImageClicked)
        self.ui.analyze_image_btn.clicked.connect (self.analyzeImageClicked)
        self.ui.change_data_btn.clicked.connect (self.changeDataClicked)
        self.ui.train_btn.clicked.connect (self.trainClicked)
        self.ui.actionLoad.triggered.connect (self.loadFile)
        self.ui.actionSave.triggered.connect (self.saveFile)
        self.ui.actionQuit.triggered.connect (self.quitProgram)
        
        self.image_model = ImagesModel (self)
        self.ui.images_tbl.setModel (self.image_model)
        self.ui.images_tbl.header ().setSectionResizeMode (QtWidgets.QHeaderView.ResizeToContents)
        self.ui.images_tbl.selectionModel ().selectionChanged.connect (self.imageSelected)
        
        self.grid_layout_image = QtWidgets.QGridLayout (self.ui.image_container_w)
        self.grid_layout_image.setContentsMargins (0, 0, 0, 0)
        self.image_w = ImageWidget (self.ui.image_container_w)
        self.grid_layout_image.addWidget(self.image_w, 0, 0, 1, 1)
        
        self.image_w.rectSelected.connect (self.diceRectSelected)

        self.dnn_position = DNNPosition ()
        self.tf_session = None
        
        self.tf_hooks = [ SessionRunHook (self) ]
        self.train_thread = None
        self.snap_thread = None
        
        self.ignore_signals = False

    def setTFSession (self, session):
        self.tf_session = session

    def snapImageClicked (self):
        print ("Snap Image Clicked")
        
        if self.snap_thread != None:
            return
        
        self.ui.snap_image_btn.setEnabled (False)
        fname = os.path.join (config.cfg.base_dir, "image_test.jpg")
        self.snap_thread = SnapThread (fname)
        self.snap_thread.imageSnappedSignal.connect (self.imageSnapped)
        self.snap_thread.start ()

    def imageSnapped (self, fname):
        print ("snap finished, fname:", repr (fname))
        if fname != None and fname != "":
            print ("create image entry")
            pos = self.image_model.addNewImage (ImageEntry (fname))
            self.selectImage (pos)
            
        print ("end")
        self.snap_thread = None
        self.ui.snap_image_btn.setEnabled (True)
        
    def analyzeImageClicked (self):
        if self.train_thread != None:
            self.ui.statusbar.showMessage ("Training active", 10000)
            return
        
        print ("Analyze Image Clicked")
        
        sel_list = self.ui.images_tbl.selectionModel ().selection ().indexes ()
        row = None
        
        for s in sel_list:
            if s.row () < len (self.image_model.data_list):
                row = s.row ()
                break
        
        if row != None:
            r = self.dnn_position.eval (lambda: self.image_model.getPredictPosData (row), hooks=self.tf_hooks)
            self.diceRectSelected (QtCore.QPoint (r[0], r[1]), QtCore.QPoint (r[2], r[3]))

            print ("Result for row {}: {}".format (row, r))
        
    def changeDataClicked (self):
        print ("Change Data Clicked")

    def trainClicked (self, checked):
        if self.ignore_signals:
            return
        
        if checked:
            if self.train_thread != None:
                print ("Training still running")
                return
    
            print ("Start training")
            self.ui.analyze_image_btn.setEnabled (False)
            self.train_thread = TrainThread (lambda: self.dnn_position.train (self.image_model.getTrainingPosData,
                                                                              self.image_model.getEvalPosData, hooks=self.tf_hooks))
            self.train_thread.iterationSignal.connect (self.trainingInfo)
            self.train_thread.finishedSignal.connect (self.trainingFinished)
            self.train_thread.start ()

        else:
            if self.train_thread == None:
                print ("No training started")
                return

            print ("Stop training")
            self.ignore_signals = True
            self.ui.train_btn.setChecked (True)
            self.ignore_signals = False

            self.train_thread.abortTraining ()
            print ("Training stop triggered")
            
    def trainingInfo (self, iterations, rmse):
        self.ui.statusbar.showMessage ("Iteration {}, root mean squared error: {}".format (iterations, rmse), 10000)
        print ("Iteration {}, root mean squared error: {}".format (iterations, rmse))
        
    def trainingFinished (self):
        #self.ignore_signals = True
        self.ui.train_btn.setChecked (False)
        #self.ignore_signals = False
        self.train_thread = None
        self.ui.analyze_image_btn.setEnabled (True)
        print ("Training stopped")
        
    def diceRectSelected (self, point1, point2):
        #print ("point:", point)
        x1 = point1.x ()
        x2 = point2.x ()
        y1 = point1.y ()
        y2 = point2.y ()
        
        if x1 > x2:
            t = x1
            x1 = x2
            x2 = t
            
        if y1 > y2:
            t = y1
            y1 = y2
            y2 = t
            
        if x2 - x1 < 200:
            d = (200 - (x2 - x1)) / 2
            x1 -= d
            x2 += d
            
        if y2 - y1 < 200:
            d = (200 - (y2 - y1)) / 2
            y1 -= d
            y2 += d
            
        self.ui.dice_position_x1_sb.setValue (x1)
        self.ui.dice_position_y1_sb.setValue (y1)
        self.ui.dice_position_x2_sb.setValue (x2)
        self.ui.dice_position_y2_sb.setValue (y2)
        self.image_w.setDicePosition ((int (x1), int (y1), int (x2), int (y2)))
        
    def updateImages (self):
        pass
    
    def updateImage (self):
        self.image_w.setImage (self.img)
        
    def showImage (self, row):
        if row < len (self.image_model.data_list):
            if self.image_model.data_list[row].hasCachedData ("display_image"):
                self.img = self.image_model.data_list[row].getCachedData ("display_image")
            else:
                fn = os.path.join (self.image_base_dir, self.image_model.data_list[row].file_name)
                self.img = QtGui.QPixmap ()
                self.img.load (fn)
                if self.img.width () < self.img.height ():
                    t = QtGui.QTransform (0, -1, 1, 0, 0, 0)
                    img2 = self.img.transformed (t)
                    self.img = img2
                    
                self.image_model.data_list[row].setCachedData ("display_image", self.img)
                
            #print ("Load:", fn)
            self.ui.dice_position_x1_sb.setMaximum (self.img.width ())
            self.ui.dice_position_y1_sb.setMaximum (self.img.height ())
            self.ui.dice_position_x2_sb.setMaximum (self.img.width ())
            self.ui.dice_position_y2_sb.setMaximum (self.img.height ())
            
            self.updateImage ()
            if self.image_model.data_list[row].hasValue ():
                self.ui.dice_value_sb.setValue (self.image_model.data_list[row].value)

            if self.image_model.data_list[row].hasPos ():
                self.image_w.setDicePosition ((self.image_model.data_list[row].pos_x1,
                                               self.image_model.data_list[row].pos_y1,
                                               self.image_model.data_list[row].pos_x2,
                                               self.image_model.data_list[row].pos_y2))
                self.ui.dice_position_x1_sb.setValue (self.image_model.data_list[row].pos_x1)
                self.ui.dice_position_y1_sb.setValue (self.image_model.data_list[row].pos_y1)
                self.ui.dice_position_x2_sb.setValue (self.image_model.data_list[row].pos_x2)
                self.ui.dice_position_y2_sb.setValue (self.image_model.data_list[row].pos_y2)
            else:
                self.image_w.setDicePosition (None)

    def imageSelected (self, selected):
        #print ("img selected")
        sel_list = selected.indexes ()
        for sel in sel_list:
            row = sel.row ()
            if row < len (self.image_model.data_list):
                self.showImage (row)
                return
    
    def selectImage (self, row):
        sel_model = self.ui.images_tbl.selectionModel ()
        index1 = self.image_model.index (row, 0)
        index2 = self.image_model.index (row, ImagesModel.COLUMNS-1)
        sel_model.select (QtCore.QItemSelection (index1, index2),
                          QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Current)
        
    def loadFile (self):
        options = QtWidgets.QFileDialog.Options ()
        file, used_filter = QtWidgets.QFileDialog.getOpenFileName (self, "Load File", config.cfg.base_dir, "All Files (*);;JSON Files (*.json)", "JSON Files (*.json)", options=options)
        
        self.openFile (file)
        config.cfg.base_dir = os.path.dirname (file)
        config.cfg.save ()
        print ("Load File {}".format (file))
        
    def saveFile (self):
        print ("Save File")

    def openFile (self, filename):
        data = []
        self.image_base_dir = ""

        f = open (filename)
        s = ""
        for l in f:
            s += l
        f.close ()
        
        try:
            data = json.loads (s)
            data = [ ImageEntry (*d) for d in data ]
            logger.info ("{} Datensätze geladen.".format (len (data)))
            self.ui.statusbar.showMessage ("Loaded {} data sets from {}".format (len (data), os.path.basename (filename)), 10000)
            self.image_model.setDataList (data, os.path.dirname (filename))
            if len (data) >= 1:
                self.image_base_dir = os.path.dirname (filename)
                self.selectImage (0)
        except ValueError as e:
            logger.error ("Ladefehler: {}".format (e))
            self.image_model.setDataList ([])
            self.ui.statusbar.showMessage ("Failed to load data from {}".format (os.path.basename (filename)), 10000)

    def quitProgram (self):
        QtWidgets.qApp.exit ()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser (description='Würfelerkennung.')
    parser.add_argument ('-t', dest='training_fname', default='/home/harald/NoBackup/Würfel/W20-grau-training-test.json',
                         help='Datei mit Trainingsdaten.')
    parser.add_argument ('-v', dest='verbosity', action="count", default=0,
                         help='increase verbosity')
    #parser.add_argument ('-n', dest='new_parse_count', type=int,
    #                     default=max_count,
    #                     help='number of new entries to parse')
    #parser.add_argument ('-w', dest='wiki', action="store_true",
    #                     help='create Wiki output format')
    #parser.add_argument ('-o', dest='out_file_name',
    #                     help='destination file instead of stdout')

    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbosity >= 2:
        log_level = logging.DEBUG
    elif args.verbosity == 1:
        log_level = logging.INFO

    log_handler = logging.StreamHandler ()
    log_handler.setLevel (log_level)

    logging.basicConfig (level=log_level, handlers=[])

    logger.addHandler (log_handler)
    logger.setLevel (log_level)

    app = QtWidgets.QApplication (sys.argv)
    window = MainWindow ()
    window.show ()

    app.exec ()
