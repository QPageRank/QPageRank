import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Polygon
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import os
import sys

sys.path.append(os.getcwd())
from Tool.JsonTool import read, save


# ---------------------- UTM CRS匹配函数（无修改） ----------------------
def get_utm_crs(gdf):
    bounds = gdf.total_bounds
    lon_cent = (bounds[0] + bounds[2]) / 2
    lat_cent = (bounds[1] + bounds[3]) / 2

    if gdf.crs.is_projected:
        return gdf.crs

    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=bounds[0],
            south_lat_degree=bounds[1],
            east_lon_degree=bounds[2],
            north_lat_degree=bounds[3],
        ),
    )

    if not utm_crs_list:
        utm_zone = int(np.floor((lon_cent + 180) / 6) + 1)
        epsg_code = 32600 + utm_zone if lat_cent >= 0 else 32700 + utm_zone
        return CRS.from_epsg(epsg_code)

    for crs in utm_crs_list:
        is_northern = getattr(crs, "utm_northern", getattr(crs, "is_northern", False))
        if (lat_cent >= 0 and is_northern) or (lat_cent < 0 and not is_northern):
            return CRS.from_epsg(crs.epsg_code)

    return CRS.from_epsg(32650) if lat_cent >= 0 else CRS.from_epsg(32750)


# ---------------------- 网格生成函数（无修改） ----------------------
def create_grid(gdf, n_rows=10, n_cols=10):
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) / n_cols
    dy = (maxy - miny) / n_rows

    grids = []
    grid_ids = []
    areas = []
    for i in range(n_rows):
        for j in range(n_cols):
            x1 = minx + j * dx
            y1 = miny + i * dy
            x2 = minx + (j + 1) * dx
            y2 = miny + (i + 1) * dy
            poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            grids.append(poly)
            grid_ids.append(f"{i}-{j}")
            areas.append(poly.area)

    grid_gdf = gpd.GeoDataFrame(
        data={"grid_id": grid_ids, "area": areas}, geometry=grids, crs=gdf.crs
    )
    return grid_gdf


# ---------------------- 网格内道路长度计算函数（无修改） ----------------------
def calculate_road_length_in_grid(road_gdf, grid_gdf):
    road_gdf = road_gdf.explode(index_parts=False).reset_index(drop=True)
    road_gdf["road_id"] = range(len(road_gdf))

    try:
        joined = gpd.sjoin(
            grid_gdf,
            road_gdf[["road_id", "geometry"]],
            how="left",
            predicate="intersects",
        )
    except TypeError:
        joined = gpd.sjoin(
            grid_gdf, road_gdf[["road_id", "geometry"]], how="left", op="intersects"
        )

    def _calc_grid_road_length(group):
        grid_geom = group.geometry.iloc[0]
        road_ids = group["road_id"].dropna().unique()
        if len(road_ids) == 0:
            return 0.0
        grid_roads = road_gdf[road_gdf["road_id"].isin(road_ids)]
        road_union = grid_roads.geometry.unary_union
        intersection = grid_geom.intersection(road_union)
        return (
            intersection.length
            if isinstance(intersection, (shapely.LineString, shapely.MultiLineString))
            else 0.0
        )

    grid_lengths = (
        joined.groupby("grid_id")
        .apply(_calc_grid_road_length)
        .reset_index(name="road_length")
    )
    grid_gdf = grid_gdf.merge(grid_lengths, on="grid_id", how="left")
    grid_gdf["road_length"] = grid_gdf["road_length"].fillna(0.0)

    return grid_gdf


# ---------------------- 主评价函数（核心新增：密度分布相似性计算） ----------------------
def grid_density_evaluation_shp(
    orig_shp_path, sub_shp_path, n_rows=10, n_cols=10, save_grid_shp=None
):
    """
    基于SHP格式路网的分块密度评价主函数
    新增：皮尔逊相关系数r、决定系数R²（密度分布相似性），贴合「密多保、疏少保」选取原则
    """
    # 步骤1：读取SHP并统一坐标系
    try:
        orig_gdf = gpd.read_file(orig_shp_path)
        sub_gdf = gpd.read_file(sub_shp_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"SHP文件不存在：{e}")
    except Exception as e:
        raise ValueError(f"读取SHP失败：{e}")

    # 检查几何类型
    valid_types = ["LineString", "MultiLineString"]
    if (
        not orig_gdf.geometry.is_valid.all()
        or not orig_gdf.geometry.type.isin(valid_types).all()
    ):
        raise TypeError(f"原始路网SHP必须为有效线要素（{','.join(valid_types)}）")
    if (
        not sub_gdf.geometry.is_valid.all()
        or not sub_gdf.geometry.type.isin(valid_types).all()
    ):
        raise TypeError(f"选取后路网SHP必须为有效线要素（{','.join(valid_types)}）")

    # 统一转换为UTM投影
    if orig_gdf.crs is None:
        raise ValueError("原始路网SHP无坐标系信息，请为SHP定义坐标系")
    orig_gdf_utm = orig_gdf.to_crs(get_utm_crs(orig_gdf))
    sub_gdf_utm = sub_gdf.to_crs(orig_gdf_utm.crs)

    # 步骤2：生成网格
    grid_gdf = create_grid(orig_gdf_utm, n_rows, n_cols)
    total_grid_count = n_rows * n_cols

    # 步骤3：计算原始/选取后路网的网格长度+密度（显式重命名）
    orig_grid_gdf = calculate_road_length_in_grid(orig_gdf_utm, grid_gdf.copy())
    orig_grid_gdf["orig_density"] = orig_grid_gdf["road_length"] / orig_grid_gdf["area"]
    orig_grid_gdf.rename(columns={"road_length": "orig_road_length"}, inplace=True)

    sub_grid_gdf = calculate_road_length_in_grid(sub_gdf_utm, grid_gdf.copy())
    sub_grid_gdf["sub_density"] = sub_grid_gdf["road_length"] / sub_grid_gdf["area"]
    sub_grid_gdf.rename(columns={"road_length": "sub_road_length"}, inplace=True)

    # 步骤4：合并网格数据
    orig_merge = orig_grid_gdf[
        ["grid_id", "geometry", "area", "orig_road_length", "orig_density"]
    ]
    sub_merge = sub_grid_gdf[["grid_id", "sub_road_length", "sub_density"]]
    eval_grid_gdf = orig_merge.merge(sub_merge, on="grid_id", how="left")

    # 步骤5：计算单网格密度保留率
    def cal_retain_rate(orig_d, sub_d):
        if orig_d < 1e-10:
            return 1.0 if sub_d < 1e-10 else 0.0
        rate = sub_d / orig_d
        return np.clip(rate, 0.0, 1.0)

    eval_grid_gdf["retain_rate"] = eval_grid_gdf.apply(
        lambda x: cal_retain_rate(x["orig_density"], x["sub_density"]), axis=1
    )

    # 步骤6：计算全局评价指标（核心新增：密度分布相似性（皮尔逊r+R²））
    valid_grid_gdf = eval_grid_gdf[
        eval_grid_gdf["orig_density"] > 1e-10
    ]  # 仅有效网格参与计算
    valid_grid_count = len(valid_grid_gdf)
    # 初始化指标
    mean_retain_rate = 0.0
    std_retain_rate = 0.0
    pearson_corr = 0.0  # 皮尔逊相关系数（密度相似性核心）
    r_squared = 0.0  # 决定系数（辅助相似性）

    if valid_grid_count >= 2:  # 至少2个有效网格才计算相关系数（否则无意义）
        # 提取有效网格的原始密度和选取后密度
        orig_densities = valid_grid_gdf["orig_density"].values
        sub_densities = valid_grid_gdf["sub_density"].values
        # 计算皮尔逊相关系数
        pearson_corr = np.corrcoef(orig_densities, sub_densities)[0, 1]
        # 计算决定系数R²（取平方，确保非负）
        r_squared = np.square(pearson_corr)
        # 原有保留率指标
        mean_retain_rate = valid_grid_gdf["retain_rate"].mean()
        std_retain_rate = valid_grid_gdf["retain_rate"].std()
    elif valid_grid_count == 1:  # 仅1个有效网格，相似性无意义，仅计算保留率
        mean_retain_rate = valid_grid_gdf["retain_rate"].mean()
        std_retain_rate = 0.0

    # 全局指标整合（新增pearson_corr、r_squared）
    global_metrics = {
        "mean_retain_rate": float(mean_retain_rate),  # 原有：全局密度保留均值
        "std_retain_rate": float(std_retain_rate),  # 原有：全局密度保留标准差
        "pearson_corr": float(
            pearson_corr
        ),  # 新增：密度分布皮尔逊相关系数（核心相似性）
        "r_squared": float(r_squared),  # 新增：决定系数（辅助相似性）
        "valid_grid_count": valid_grid_count,  # 原有：有效网格数
        "total_grid_count": total_grid_count,  # 原有：总网格数
        "crs": orig_gdf_utm.crs.to_wkt(),  # 原有：计算坐标系
    }

    # 步骤7：保存网格结果（可选）
    if save_grid_shp:
        try:
            save_gdf = eval_grid_gdf[
                ["grid_id", "geometry", "orig_density", "sub_density", "retain_rate"]
            ]
            save_gdf.to_file(save_grid_shp, encoding="utf-8")
            print(f"网格评价结果已保存为SHP：{save_grid_shp}")
        except Exception as e:
            print(f"保存SHP失败：{e}")

    # 步骤8：构造返回结果
    grid_details = eval_grid_gdf.set_index("grid_id")[
        [
            "orig_density",
            "sub_density",
            "retain_rate",
            "orig_road_length",
            "sub_road_length",
            "area",
        ]
    ].to_dict(orient="index")

    return {
        "grid_details": grid_details,
        "global_metrics": global_metrics,
        "grid_gdf": eval_grid_gdf,
        "valid_grid_gdf": valid_grid_gdf,  # 新增返回有效网格，方便后续可视化
    }


def run_density(index):
    cfg = read(os.getcwd() + "/config.json")
    density_path = (
        os.getcwd()
        + cfg["roadselect"]["preprocessing"]["basicpath"]
        + "/"
        + str(index)
        + "/Result/pearson_density.json"
    )
    ORIG_SHP = (
        os.getcwd() + "/Data/RoadSelect/" + str(index) + "/sample/road_sample.shp"
    )
    result_map = {}
    for key in ["quantum", "classic", "GNN"]:
        SUB_SHP = (
            os.getcwd()
            + "/Data/RoadSelect/"
            + str(index)
            + "/SelectResult/"
            + key
            + "_0.4_0.85_0.shp"
        )
        # 执行评价（建议网格数根据路网范围调整，如20×20）
        result = grid_density_evaluation_shp(
            orig_shp_path=ORIG_SHP,
            sub_shp_path=SUB_SHP,
            n_rows=10,
            n_cols=10,
        )
        pearson_corr = result["global_metrics"]["pearson_corr"]
        result_map[key] = pearson_corr
    save(density_path, result_map)


if __name__ == "__main__":
    index = 116
    run_density(index)
