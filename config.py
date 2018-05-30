#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import os
import json

class Config (object):
    def __init__ (self):
        self.data = {"base_dir": "",
                     "capture_command": "raspistill -o -" }

        if "HOME" in os.environ:
            self.data["base_dir"] = os.environ["HOME"]

    def __setattr__(self, *args, **kwargs):
        if len(args) >= 2:
            if args[0] != "data" and args[0] in self.data:
                self.data[args[0]] = args[1]
        
        return object.__setattr__(self, *args, **kwargs)
            
    def __getattribute__(self, *args, **kwargs):
        if len(args) >= 1:
            if args[0] != "data" and args[0] in self.data:
                return self.data[args[0]]
        
        return object.__getattribute__(self, *args, **kwargs)

    @staticmethod
    def defaultFileName ():
        dir = ""
        if "HOME" in os.environ:
            dir = os.environ["HOME"]
            
        return os.path.join (dir, ".dice.cfg")
        
    def save (self, fname = None):
        if fname == None:
            fname = Config.defaultFileName ()
            
        f = open (fname, "w")
        json.dump (self.data, f, indent=2, sort_keys=True)
        f.close ()
        
    def load (self, fname = None):
        if fname == None:
            fname = Config.defaultFileName ()
            
        data = None
        if os.path.isfile (fname):
            f = open (fname, "r")
            try:
                data = json.load (f)
            except Exception as e:
                print ("Error:", e)
                data = None
                
            f.close ()
            
        if not isinstance (data, dict):
            print ("failed:", repr (data))
            return
        
        for k in data:
            print ("key:", repr (k), "value:", repr (data[k]))
            if k in self.data:
                self.data[k] = data[k]


cfg = Config ()
cfg.load ()

