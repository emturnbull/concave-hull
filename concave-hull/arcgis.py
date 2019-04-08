import geom
import tweet
import sys
import arcpy
import os
import importlib

importlib.reload(geom)
 
def init_tweet_arcgis(debug_mode=False, log_file=None):
    tweet.init_echo(False)
    min_level = tweet.Printer.INFO
    if debug_mode:
        min_level = tweet.Printer.DEBUG
    tweet.Printer().register_python_console(min_level)
    tweet.Printer().register_printer_function(arcpy.AddMessage, min_level,tweet.Printer.INFO)
    tweet.Printer().register_printer_function(arcpy.AddWarning, tweet.Printer.WARNING, tweet.Printer.WARNING)
    tweet.Printer().register_printer_function(arcpy.AddError, tweet.Printer.ERROR)
    if not log_file is None:
        tweet.Printer().register_log_file(log_file, min_level)
    tweet.echo(len(tweet.Printer().instance.printers))

init_tweet_arcgis(False, None)

tweet.debug("In script {}".format(sys.argv[0]))

"""
    Input: points feature class, grouping field (optional), etc
    Output: concave polygons
"""

testing = False
if testing:
    point_features = r'C:\Students\101056654\lab4\Lab4.gdb\Proximity_Check2004_CADToGEo\Point' # feature class
    group_field = '' # grouping field
    iteration_cap = None # iteration cap
    minimum_angle = 0 # angle
    minimum_length = 0 # edge distance
    output_geodatabase = r'C:\Students\101056654\lab4\Lab4.gdb\\'
    output_class = 'test6'
else:
    point_features = sys.argv[1]
    group_field = sys.argv[2]
    if not group_field or group_field == '' or group_field == '#':
        group_field = None
    new_fc = sys.argv[3]
    output_geodatabase = os.path.dirname(new_fc)
    output_class = os.path.basename(new_fc)
    if sys.argv[4] == '#':
        iteration_cap = 0
    else:
        iteration_cap = float(sys.argv[4])
        if iteration_cap is None or iteration_cap <= 0:
            iteration_cap = None
    minimum_angle = float(sys.argv[5])
    if minimum_angle <= 0:
        minimum_angle = 0
    if minimum_angle >= 180 or minimum_angle == '#':
        minimum_angle = 180
    minimum_length = float(sys.argv[6])
    if minimum_length <= 0 or minimum_length == '#':
        minimum_length = 0

fields = ['SHAPE@XY']
tweet.debug(str(group_field))
if not group_field is None:
    fields.append(group_field)
point_lists = {}
with arcpy.da.SearchCursor(point_features, fields, explode_to_points=True) as cursor:
    for row in cursor:
        group_by = 'all'
        if not group_field is None:
            group_by = row[1]
        if not group_by in point_lists:
            point_lists[group_by] = []
        point_lists[group_by].append(row[0])


spatialRef = arcpy.Describe(point_features).spatialReference

if not arcpy.Exists(new_fc):
    tweet.info("Creating feature class {}".format(new_fc))
    result = arcpy.management.CreateFeatureclass(output_geodatabase, output_class, "POLYGON", spatial_reference=spatialRef)
    if not group_field is None:
        arcpy.AddField_management(new_fc, group_field, "TEXT")
else:
    tweet.info("Emptying feature class {}".format(new_fc))
    arcpy.DeleteFeatures_management(new_fc)

insert = ['SHAPE@']
if not group_field is None:
    insert.append(group_field)

with arcpy.da.InsertCursor(new_fc, insert) as icursor:
    for group in point_lists:
        points = point_lists[group]
        convex_hull = geom.convex_hull(points)
        concave_hull = geom.concave_hull(convex_hull, points, min_length_fraction=minimum_length, min_angle=minimum_angle, max_iterations=iteration_cap)
        if not group_field is None:
            icursor.insertRow([concave_hull, group])
        else:
            icursor.insertRow([concave_hull])


   


    
    
        