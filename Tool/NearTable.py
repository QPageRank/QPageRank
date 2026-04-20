# -*- coding: utf-8 -*-
import os
import sys
import arcpy
import csv
import json
import chardet
import codecs


def read_config(filepath):
    encoding = detect_encoding(filepath)
    with codecs.open(filepath, "r", encoding=encoding) as f:
        return json.load(f)


def detect_encoding(path):
    with open(path, "rb") as f:
        raw = f.read()
    result = chardet.detect(raw)
    return result.get("encoding", "utf-8") or "utf-8"


def neartable(gdb_path, in_features, near_features, out_table, stroke_txt):
    arcpy.env.workspace = gdb_path
    arcpy.env.overwriteOutput = True

    # Search radius and closest count
    search_radius = 0
    closest_count = 0  # no limit

    # Generate Near Table
    arcpy.GenerateNearTable_analysis(
        in_features,
        near_features,
        out_table,
        search_radius,
        "NO_LOCATION",
        "NO_ANGLE",
        "ALL",
        closest_count,
    )

    print("Near table generated successfully.")

    # Get fields and remove 4th column
    fields = arcpy.ListFields(out_table)
    field_names = [f.name for f in fields]

    print("Original fields:", field_names)

    if len(field_names) > 3:
        removed_field = field_names.pop(3)
        print("Removed 4th field:", removed_field)
    else:
        print("Less than 4 fields, skip removing any field.")

    # Export path
    # stroke_txt = r"D:\python\qpage-rank\Data\RoadSelect\NearTable_Output.txt"

    # Export to txt file with comma delimiter (Python 2.7)
    with open(stroke_txt, "wb") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(field_names)

        with arcpy.da.SearchCursor(out_table, field_names) as cursor:
            for row in cursor:
                writer.writerow(row)

    print("Export completed. File saved at: {}".format(stroke_txt))


def run_NearTable(index):
    configpath = os.getcwd() + "/config.json"
    config = read_config(configpath)
    gdb_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Roadgdbdata/"
        + str(index)
        + ".gdb"
    )
    stroke_txt = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/stroke"
        + str(index)
        + ".txt"
    )
    layer_name = "road_" + str(index)

    # Input and output
    in_features = layer_name
    near_features = layer_name
    out_table = "NearTable_Output"
    neartable(gdb_path, in_features, near_features, out_table, stroke_txt)


if __name__ == "__main__":
    import sys

    run_NearTable(sys.argv[1])
