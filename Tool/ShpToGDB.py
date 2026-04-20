# ShpToGDB.py (Python 2.7)
# -*- coding: utf-8 -*-
import arcpy
import os
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


def change(input_shp, output_gdb, output_fc_name):
    # ensure parent folder exists
    folder = os.path.dirname(output_gdb)
    gdbname = os.path.basename(output_gdb)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # create GDB if absent
    if not arcpy.Exists(output_gdb):
        arcpy.CreateFileGDB_management(out_folder_path=folder, out_name=gdbname)

    # full output path
    out_fc_path = os.path.join(output_gdb, output_fc_name)

    # delete existing feature class
    if arcpy.Exists(out_fc_path):
        arcpy.Delete_management(out_fc_path)

    # convert
    arcpy.conversion.FeatureClassToFeatureClass(
        in_features=input_shp, out_path=output_gdb, out_name=output_fc_name
    )
    return out_fc_path


def export_table_to_txt(feature_class, output_txt, delimiter=","):
    txt_folder = os.path.dirname(output_txt)
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
    fields = [f.name for f in arcpy.ListFields(feature_class) if f.type != "Geometry"]
    with codecs.open(output_txt, "w", encoding="utf-8") as txt:
        txt.write(delimiter.join(fields) + "\n")
        with arcpy.da.SearchCursor(feature_class, fields) as cur:
            for row in cur:
                vals = [str(v) if v is not None else "" for v in row]
                txt.write(delimiter.join(vals) + "\n")


def run_ShpToGDB(index):

    configpath = os.getcwd() + "/config.json"
    config = read_config(configpath)
    out_shp = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/stroke/"
        + "/road_stroke"
        + ".shp"
    )
    gdb_path = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Roadgdbdata/"
        + str(index)
        + ".gdb"
    )
    nodedata_txt = (
        os.getcwd()
        + config["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/txt/nodedata"
        + str(index)
        + ".txt"
    )
    layer_name = "road_" + str(index)
    change(out_shp, gdb_path, layer_name)

    # conversion
    fc_full = change(out_shp, gdb_path, layer_name)

    export_table_to_txt(fc_full, nodedata_txt)


# run_ShpToGDB("1")
if __name__ == "__main__":
    import sys

    run_ShpToGDB(sys.argv[1])
