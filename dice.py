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

def cnn_model_fn (features, labels, mode, params):
  """Model function for CNN."""
  # Input Layer
  layers = params.get ("layers", [6348, 2048, 512, 128, 32, 2])

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

def loadSmallImage (fname, *label):
    #print ("###### fname:", fname)
    image_string = tf.read_file (fname)
    image = tf.image.decode_jpeg (image_string)
    image = tf.image.convert_image_dtype (image, dtype=tf.float32)
    image = tf.image.resize_images (image, [92, 69])
    image = tf.image.rgb_to_grayscale (image)
    image = tf.reshape (image, (6348, ))
    
    label = [ tf.cast (l, dtype=tf.float32) for l in label ]
    #label = tf.cast (label, dtype=tf.float32)

    #print ("###### img shape:", image.shape)
    #print ("###### image:", image)
    #print ("###### label:", label)
    #print ("###### img size:", tf.size (image_resized).eval ())
    return ( {"images_small": image }, *label )
    #return ( {"image_small": image_resized, "image": image_decoded}, ( nr, pos ) )

def loadData (fname):
    f = open (fname)
    s = ""
    for l in f:
        s += l
    f.close ()

    data = []
    try:
        data = json.loads (s)
        logger.info ("{} Datensätze geladen.".format (len (data)))
    except ValueError as e:
        logger.error ("Ladefehler: {}".format (e))
        data = []
    
    filenames = [ fn for fn, nr, posx, posy in data ]
    numbers = [ nr for fn, nr, posx, posy in data ]
    pos = [ (posx, posy) for fn, nr, posx, posy in data ]

    dataset = tf.data.Dataset.from_tensor_slices ( (filenames, pos) )
    dataset = dataset.map (loadSmallImage)
    #dataset = dataset.shuffle (buffer_size=1)
    dataset = dataset.batch (10)
    dataset = dataset.repeat (10)
    
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels

    print (dataset)
    print (dataset)
    
    return ({'images_small': dataset}, labels)

class Trainer:
    def __init__ (self, training_fname):
        self.training_fname = training_fname
        
    def train (self):
        #self.training_data = loadData (self.training_fname)
        #print (self.training_data)
        
        image_small = tf.feature_column.numeric_column ('images_small',
                                                        #shape = (1, 92*69),
                                                        dtype=tf.float32)

        estimator = tf.estimator.Estimator(
                        model_fn=cnn_model_fn,
                        params={
                            "layers": [6348, 2048, 512, 128, 32, 2],
                            "learning_rate": 0.001,
                            "optimizer": tf.train.AdamOptimizer
                        },
                        model_dir="/home/harald/NoBackup/Würfel/ModelDir")

        #estimator = tf.estimator.DNNRegressor (
        #    feature_columns = [image_small],
        #    hidden_units=[8192, 2048, 512, 128, 32],
        #    label_dimension=2,
        #    model_dir="/home/harald/NoBackup/Würfel/ModelDir")
        
        while True:
            estimator.train (input_fn=lambda: loadData (self.training_fname), steps=2)
            r = estimator.evaluate (input_fn=lambda: loadData (self.training_fname), steps=1)
            print ("steps: {} loss: {} rmse: {}".format (r['global_step'], r['loss'], r['rmse']))
            

class ImagesModel (QtGui.QStandardItemModel):
    
    def __init__ (self):
        QtGui.QStandardItemModel.__init__ (self, 5, 3)
        self.data_list = ["Test"]
        #self.setData (self.index (0, 0), "Test")
        self.setData (self.index (0, 0), "Test")
        #print (dir (self))
        
    def setDataList (self, data):
        self.beginResetModel ()
        self.data_list = data
        self.endResetModel ()
        self.dataChanged.emit (self.index (0,0), self.index (len (data)-1, 3))
    
    def columnCount (self, parent_index):
        if parent_index == QtCore.QModelIndex ():
            return 3
        
        return 0
    
    def headerData (self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal:
            if role == QtCore.Qt.DisplayRole:
                if section == 0:
                    return "Image"
                
                if section == 1:
                    return "Pos"
                
                return "Value"

    def rowCount (self, parent_index):
        if parent_index == QtCore.QModelIndex ():
            return len (self.data_list)

        return 0
    
    def data (self, index, role):
        if role == QtCore.Qt.DisplayRole:
            print ("index:", index.row (), index.column ())
            #print ("args:", repr (args))
            #print ("index", repr (index))
            #print ("role ", repr (role))
            return "Blub"
        
        return None

class MainWindow (QtWidgets.QMainWindow):
    ui = Ui_main_window ()
    
    def __init__ (self):
        QtWidgets.QMainWindow.__init__ (self)
        self.ui = Ui_main_window ()
        self.ui.setupUi (self)
        self.ui.snap_image_btn.clicked.connect (self.snapImageClicked)
        self.ui.analyze_image_btn.clicked.connect (self.analyzeImageClicked)
        self.ui.add_data_btn.clicked.connect (self.addDataClicked)
        self.ui.actionLoad.triggered.connect (self.loadFile)
        self.ui.actionSave.triggered.connect (self.saveFile)
        
        self.image_model = ImagesModel ()
        self.ui.images_tbl.setModel (self.image_model)
    
    def snapImageClicked (self):
        print ("Snap Image Clicked")
        
    def analyzeImageClicked (self):
        print ("Analyze Image Clicked")
        
    def addDataClicked (self):
        print ("Add Data Clicked")
        
    def updateImages (self):
        pass
        
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

        f = open (filename)
        s = ""
        for l in f:
            s += l
        f.close ()
        
        try:
            data = json.loads (s)
            logger.info ("{} Datensätze geladen.".format (len (data)))
            self.ui.statusbar.showMessage ("Loaded {} data sets from {}".format (len (data), os.path.basename (filename)), 10000)
            self.image_model.setDataList (data)
        except ValueError as e:
            logger.error ("Ladefehler: {}".format (e))
            self.image_model.setDataList ([])
            self.ui.statusbar.showMessage ("Failed to load data from {}".format (os.path.basename (filename)), 10000)

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
    #trainer = Trainer (args.training_fname)
    #trainer.train ()
