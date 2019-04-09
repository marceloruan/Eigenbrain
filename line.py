#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  line.py
#  
#  Copyright 2019 Marcelo Ruan <marcelo@marcelo-HP-Pavilion-14-Notebook-PC>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import csv
from pathlib import Path
import os
destino = "/home/marcelo/70-74.9/F"

with open("70-74.9.csv") as f:
	reader = csv.reader(f)
	for row in reader:
		call = "mv " +row[0]+ " " +destino
		os.system(call)
		#print(row[0])
		#Path('%s'%(row[0])).touch()

