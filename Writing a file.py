# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:13:43 2018

@author: TKS
"""

import csv



fields = ['Name', 'Branch', 'Year', 'CGPA']



rows = [ ['Nikhil', 'COE', '2', '9.0'],
		['Sanchit', 'COE', '2', '9.1'],
		['Aditya', 'IT', '2', '9.3'],
		['Sagar', 'SE', '1', '9.5'],
		['Prateek', 'MCE', '3', '7.8'],
		['Sahil', 'EP', '2', '9.1']]


filename = "C://Users//TKS//Desktop//data//university_records.csv"


with open(filename, 'w') as csvfile:
	# creating a csv writer object
	csvwriter = csv.writer(csvfile)
	
	# writing the fields
	csvwriter.writerow(fields)
	
	# writing the data rows
	csvwriter.writerows(rows)
