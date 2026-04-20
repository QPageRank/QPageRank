### English Translation of QPageRank Road Selection System README  

# QPageRank Road Selection System README  

## 1. Project Overview  
This project is designed for automatic road network selection and comparative evaluation. Its core objective is to convert raw road data into a network structure suitable for graph algorithm processing, then run three types of methods for road importance ranking and selection:  

1. **Classic PageRank**  
2. **Quantum PageRank**  
3. **GNN (GraphSAGE) baseline method**  

Subsequently, the results of these three methods are converted into road layers, and evaluated against metrics such as coverage rate, connectivity, and density.  

In terms of code structure, the project includes the following phases:  
- Raw road data preprocessing  
- Road network stroke aggregation and dual graph construction  
- Classic PageRank calculation  
- Quantum PageRank circuit and matrix evolution calculation  
- GNN training and inference  
- Export of road selection results to Shapefile  
- Comparative evaluation and result statistics  

---  

## 2. Runtime Environment Instructions  

### 2.1 Python Version  
The project uses two runtime environments:  
- **Python 3 environment**: The main workflow, Quantum PageRank, GNN, and most data processing run here.  
- **Python 2.7 + ArcPy / ArcGIS**: Used for partial GIS tool scripts (e.g., `Tool/ShpToGDB.py`, `Tool/NearTable.py`).  

The `python2path` in `config.json` must point to the Python 2.7 interpreter bundled with ArcGIS, for example:  

```json
"python2path": "D:/Python27/ArcGIS10.2/python.exe"
```  

### 2.2 Key Dependencies  
Common dependencies in the project include:  
- `numpy`  
- `scipy`  
- `pandas`  
- `networkx`  
- `geopandas`  
- `shapely`  
- `fiona`  
- `pyproj`  
- `chardet`  
- `h5py`  
- `tqdm`  
- `scikit-learn`  
- `torch`  
- `torch_geometric`  
- `qiskit`  
- `arcpy` (ArcGIS environment)  

### 2.3 Working Directory Requirements  
The project code extensively uses `os.getcwd()` to concatenate paths. Therefore, we recommend:  
- Setting the project root directory as the current working directory when running the code  
- Keeping `config.json` in the same directory as the code root  
- Maintaining the directory structure of `Data/`, `Tool/`, `RoadSelect/`, `Preprocessing/`, etc.  

---  

## 3. Overall Operation Workflow  
Below is the recommended complete execution sequence.  

### Step 1: Prepare Raw Data  
Place raw road vector data in the corresponding directory `Data/RoadSelect/{index}/`. This typically includes:  
- Raw road shapefile or gdb  
- Stroke results  
- nodedata/stroke text files  
- Subsequent result directories  

### Step 2: Road Preprocessing  
The preprocessing phase primarily accomplishes three tasks:  
1. **Stroke aggregation**: Merge road polylines into longer strokes based on angle rules  
2. **Shp to GDB conversion**: Convert stroke/raw roads to File Geodatabase for ArcPy operations  
3. **Near Table generation**: Construct spatial neighbor relationships between roads and export to text files  

Outputs of these steps are generally written to:  
- `Data/RoadSelect/{index}/stroke/`  
- `Data/RoadSelect/{index}/Roadgdbdata/`  
- `Data/RoadSelect/{index}/txt/`  

### Step 3: Construct Probability Transition Matrix  
Both Quantum PageRank and Classic PageRank rely on transition matrices. The code constructs the following based on `stroke.txt` and `nodedata.txt`:  
- Probability transition matrix `ProbabilityMatrix`  
- Node attribute dictionary `node_att`  

### Step 4: Run Classic PageRank  
Classic PageRank iterates directly on the probability transition matrix to obtain ranking values for each road, outputting to:  
- `Result/PageRankWeightResult.json`  

### Step 5: Run Quantum PageRank  
The Quantum PageRank workflow is longer and generally includes:  
1. Construct initial state  
2. Construct U operator (projection, swap, evolution matrices, etc.)  
3. Iteratively apply the U operator  
4. Read HDF5 result blocks and restore to complete ranking results  
5. Output to `Result/QPageRankWeightResult.json`  

### Step 6: Run GNN Baseline  
The GNN script:  
1. Constructs a dual graph from training data  
2. Trains GraphSAGE  
3. Performs inference on the test graph  
4. Aggregates node scores into road scores  
5. Outputs to `Result/GNNResult.json`  

### Step 7: Select Roads by Ratio  
`SelectRate.py` selects the top N roads from the three types of results according to a given ratio (e.g., 0.2, 0.4, 0.6) and outputs Shapefiles to:  
- `Data/RoadSelect/{index}/SelectResult/`  

### Step 8: Comparative Evaluation  
The project provides multiple evaluation scripts:  
- `RoadMatch.py`: Road segment overlap/coverage evaluation  
- `Connectivity.py`: Connectivity change evaluation  
- `Caldensity.py`: Density/spatial distribution evaluation  

Final outputs include:  
- `cover.json`  
- `connectivity.json`  
- `pearson_density.json`  

---  

## 4. Recommended Execution Entry Points  
The main entry point of the project is located at:  

```text
Engine/QPageRankEngine.py
```  

This file is responsible for orchestrating the entire workflow. In the current version, multiple steps are retained as comments in `main()`, making it easy to enable or disable them on demand. It functions more as a "debugging master control script" rather than a fixed production entry point.  

We recommend enabling the corresponding steps one by one in the following order:  

```python
# Preprocessing
run_stroke(index)
subprocess.check_call([python2_path, shptogdb_path, str(index)])
subprocess.check_call([python2_path, neartable_path, str(index)])

# Quantum PageRank
run_CreatStartEngine(index)
run_CreatUEngine(index)
run_ApplyUEngine(index)
run_ResultProcess(index)

# Classic PageRank
run_ClassicEngine(index)

# GNN
runGNN(index)

# Result selection and evaluation
run_SelectRate(index)
run_RoadMatch(index)
run_R_path(index)
run_density(index)
```  

---  

## 5. `config.json` Configuration Instructions  
`config.json` is the core configuration file of the entire system.  

### 5.1 Global Configuration  
- `noise`: List of noise levels for quantum/classic/GNN comparison  
- `evolution_way`: Encoding parameters for quantum evolution methods  
- `block_num`: Number of blocks or cache size for block matrices  
- `node_num`: Number of road network nodes (dynamically written back during runtime)  

### 5.2 Preprocessing Configuration  
- `orignosmroadpath`: Path to raw road data  
- `basicpath`: Root directory for project data (typically `/Data/RoadSelect`)  

### 5.3 Initial State Construction Configuration  
- `block_size`: Block size when creating the initial state  
- `alpha`: Damping factor or related weight  

### 5.4 U Operator Configuration  
- `proj_output_hdf5_filename`: Output path for projection operator  
- `swap_output_hdf5_filename`: Output path for swap operator  
- `u_output_hdf5_filename`: Prefix for final U operator output  
- `temp_path`: Directory for intermediate temporary files  
- `max_cache_blocks`: Upper limit for the number of cache blocks  
- `alpha`: Parameter used in evolution  

### 5.5 ApplyU Configuration  
- `next_path`: Temporary file path for the next state  
- `pre_path`: Temporary file path for the previous state  
- `output_path`: Prefix for final result output  
- `max_iterations`: Maximum number of iterations  
- `tolerance`: Convergence threshold  
- `alphaway`: Alpha processing method  
- `alpha_s`: List of alpha values to iterate over  

### 5.6 Result Processing Configuration  
- `accpath`: Path for accuracy-related results  
- `roadshowpath`: Path for road gdb used for visualization  
- `layername`: Layer name  
- `selectshppath`: Path for selected result shp files  
- `zoomratio`: List of road selection ratios  

### 5.7 Classic PageRank Configuration  
- `max_iterations`: Number of iterations  
- `tolerance`: Convergence threshold  
- `alpha`: List of damping factors  

### 5.8 GNN Configuration  
- `train_format`: Format of training data  
- `test_format`: Format of test data  
- `train_layer`: Training layer  
- `test_layer`: Test layer  
- `gnnresult`: Path to save GNN results  

---  

## 6. Directory Structure Instructions  

### 6.1 Top-Level Directories  
- `Engine/`: Master control entry scripts  
- `Preprocessing/`: Preprocessing modules  
- `RoadSelect/`: Three types of road selection methods and evaluation modules  
- `Tool/`: Underlying utility functions and matrix operation tools  
- `Circuit/`: Independent quantum circuit construction experimental code  
- `Data/`: Test case data and output results  
- `config.json`: Global configuration file  

### 6.2 `Data/RoadSelect/{index}/` Directory  
Each `index` corresponds to a set of independent experimental data. Common subdirectories include:  
- `stroke/`: Stroke aggregation result shapefiles  
- `txt/`: Converted node/stroke text data  
- `Roadgdbdata/`: File GDB data  
- `Matrix/`: Quantum operator and intermediate matrix HDF5 files  
- `Result/`: PageRank, QPageRank, GNN, and statistical result JSON files  
- `SelectResult/`: Final selected road shapefiles  

---  

## 7. Role of Each Code File  

### 7.1 `Engine/QPageRankEngine.py`  
Master entry and workflow scheduling script for the project. It unifies preprocessing, Quantum PageRank, Classic PageRank, GNN, result filtering, and evaluation scripts. The current code contains extensive comments, making it suitable for manually enabling/disabling experimental phases.  

### 7.2 `Preprocessing/RS_model.py`  
Used for power function fitting experiments in papers. It performs logarithmic transformation and linear regression on a set of sample data using `sklearn` to fit a power-law model and outputs the goodness of fit `R²`. If your paper includes a section on "proportional relationships/empirical model fitting", this script serves as an auxiliary analysis tool.  

### 7.3 `Preprocessing/stroke/StrokeEngine.py`  
Main program for stroke aggregation. It reads raw road shapefiles, merges adjacent line segments with similar directions into strokes based on angle thresholds, and outputs a new stroke layer. This step reduces road fragmentation and provides more stable road units for subsequent dual graph and PageRank calculations.  

### 7.4 `Preprocessing/stroke/utiles.py`  
Auxiliary function library for stroke processing. It mainly includes:  
- `merge_road_points`: Merge road point sequences  
- `touchable`: Determine if roads can be connected  
- `calculate_angele`: Calculate the angle between adjacent line segments  

These functions form the underlying support for `StrokeEngine.py`.  

### 7.5 `Circuit/circuit.py`  
Independent experimental script for Quantum PageRank circuit construction. It includes:  
- Recursive calculation of rotation angles from probability distributions  
- Multi-controlled Hadamard / multi-controlled rotation gates  
- Initial state circuit  
- Reflection operator  
- Shift operator  
- Overall Quantum PageRank circuit stitching  

From the directory location, it functions more as an "independent experimental version" or "prototype version" of the circuit implementation.  

### 7.6 `RoadSelect/Quantum/Circuit.py`  
One of the official implementation versions of Quantum PageRank. Similar in functionality to `Circuit/circuit.py` above, but more tightly integrated with the project's road selection workflow.  

It is primarily responsible for:  
- Constructing the initial state from the transition matrix  
- Constructing the reflection operator  
- Constructing the shift operator  
- Combining into a complete Quantum PageRank evolution circuit  

### 7.7 `RoadSelect/Quantum/CreatProbabilityMatrix.py`  
Core class for constructing probability transition matrices from `stroke.txt` and `nodedata.txt`.  

Main functions:  
- Reading stroke and node data in text format  
- Establishing mappings between node IDs and road attributes  
- Generating the probability transition matrix `ProbabilityMatrix`  
- Generating adjacency relationships and node attribute dictionary `node_att`  

This step is the common foundation for Classic PageRank, Quantum PageRank, and partial GNN comparisons.  

### 7.8 `RoadSelect/Quantum/CreatStartEngine.py`  
Entry point for "initial state" construction in Quantum PageRank.  

It:  
- First calls `CreatProbabilityMatrix` to obtain the transition matrix  
- Normalizes initial node weights  
- Calls `Tool/CreatStart.py` to write the initial state HDF5 file  

### 7.9 `RoadSelect/Quantum/CreatUEngine.py`  
Entry point for U operator construction in Quantum PageRank.  

It:  
- Reads the probability transition matrix  
- Calls `Tool/CreatU.py`  
- Constructs projection operators, swap operators, and U operators  
- Saves results as block matrices in HDF5 format  

### 7.10 `RoadSelect/Quantum/ApplyUEngine.py`  
Entry point for quantum evolution execution.  

Its role is to:  
- Count the number of nodes and write back to `config.json`  
- Calculate the required block size  
- Read initial state and U operator HDF5 files  
- Call `Tool/ApplyU.py` for iterative evolution  
- Output result HDF5 files for different `noise` and `alpha` values  

### 7.11 `RoadSelect/Quantum/ResultProcess.py`  
Script for restoring and organizing Quantum PageRank results.  

It:  
- Reads block results from HDF5  
- Merges into a complete PageRank score vector  
- Sorts the results  
- Outputs to `QPageRankWeightResult.json`  

### 7.12 `RoadSelect/Classic/PageRankWeight.py`  
Class for Classic PageRank weight calculation.  

The core logic is straightforward:  
- Construct the Google matrix from the probability matrix  
- Iteratively update from initial node attributes  
- Record the difference in changes per iteration  
- Output ranking values  

### 7.13 `RoadSelect/Classic/PageRankWeightEngine.py`  
Workflow wrapper for Classic PageRank.  

It is responsible for:  
- Reading `stroke.txt` and `nodedata.txt`  
- Constructing the probability matrix  
- Running Classic PageRank  
- Saving ranking and numerical results to `PageRankWeightResult.json`  

### 7.14 `RoadSelect/GNN/GNN.py`  
Implementation of the GNN baseline method using GraphSAGE.  

It mainly:  
- Constructs a graph from road data  
- Splits training/validation/test sets  
- Trains a GraphSAGE classification/scoring model  
- Performs inference on the test graph  
- Aggregates node scores into road scores  
- Saves results to `GNNResult.json`  

This module serves as a comparison baseline for Classic PageRank and Quantum PageRank.  

### 7.15 `RoadSelect/Compare/SelectRate.py`  
Main script for exporting road results by selection ratio.  

It reads:  
- `PageRankWeightResult.json`  
- `QPageRankWeightResult.json`  
- `GNNResult.json`  

Then selects the top-ranked roads according to the ratios specified in `zoomratio` and finally exports them as Shapefiles.  

It also includes two export methods:  
- `outroadshp`: Export from combined Classic/Quantum/GNN results  
- `outroadshp_gnn`: Export only GNN results  

### 7.16 `RoadSelect/Compare/RoadMatch.py`  
Used for road coverage and overlap analysis.  

The main approach is to:  
- Project road data to an appropriate CRS  
- Perform matching using buffers and direction constraints  
- Count the proportion of matched roads  
- Output to `cover.json`  

These metrics are suitable for measuring whether the selected roads cover the reference road network.  

### 7.17 `RoadSelect/Compare/Connectivity.py`  
Used to evaluate changes in connectivity of road selection results.  

It constructs the original graph and subgraphs, compares changes in shortest paths or reachability efficiency between random node pairs, and writes to:  
- `connectivity.json`  

### 7.18 `RoadSelect/Compare/Caldensity.py`  
Used to evaluate road density distribution and spatial statistical characteristics.  

It:  
- Automatically selects a UTM projection  
- Constructs a grid  
- Counts road length in each grid  
- Calculates the correlation between original and sub-network densities  
- Outputs to `pearson_density.json`  

### 7.19 `Tool/JsonTool.py`  
General JSON utility.  

Functions include:  
- `read`: Read JSON files  
- `save`: Save JSON files  
- `saveresult`: Append key-value pairs to existing JSON  
- `save_match`: Save JSON with numpy data  
- `modify`: Modify `node_num` and `block_num` in `config.json`  
- `detect`: Automatically detect file encoding  

This is the most commonly used general read/write utility in the project.  

### 7.20 `Tool/MatrixMultiplication.py`  
The lowest-level and most critical block matrix calculation tool in Quantum PageRank.  

It implements:  
- Sparse matrix block read/write  
- HDF5 block cache persistence  
- Large matrix block multiplication  
- Block addition/subtraction, scalar multiplication, complex multiplication  
- Reading result blocks from HDF5 and merging into a complete matrix  

Since the matrix scale in Quantum PageRank can be extremely large, this file splits ultra-large matrix calculations into manageable block tasks.  

### 7.21 `Tool/CreatStart.py`  
Utility class for initial state matrix construction.  

It is responsible for:  
- Adding damping factors to the initial probability matrix  
- Generating initial state matrices block by block  
- Writing intermediate results to HDF5  
- Maintaining `node_num` and `block_num`  

### 7.22 `Tool/CreatU.py`  
Utility class for U operator construction.  

It is responsible for:  
- Generating projection operators  
- Generating swap operators  
- Combining into evolution U operators  
- Saving results using block matrices and HDF5  

### 7.23 `Tool/ApplyU.py`  
Utility class for iterative U operator execution.  

It is responsible for:  
- Loading the initial state  
- Reading U operators  
- Performing step-by-step iterative evolution  
- Judging convergence or reaching the maximum number of iterations  
- Saving results after each evolution  

### 7.24 `Tool/ShpToGDB.py`  
ArcPy tool script (Python 2.7).  

Functions include:  
- Converting Shapefiles to File Geodatabase  
- Exporting attribute tables to TXT  
- Preparing data for subsequent `NearTable` and other ArcGIS operations  

### 7.25 `Tool/NearTable.py`  
ArcPy tool script (Python 2.7).  

Its role is to:  
- Call `GenerateNearTable_analysis`  
- Generate road neighbor tables  
- Export to txt  

This is an important prerequisite step for dual graph construction and connectivity relationship extraction for roads.  

### 7.26 Relationship between `Circuit/` and `RoadSelect/Quantum/`  
Both directories contain quantum circuit construction logic. They can generally be understood as:  
- `Circuit/`: Focused on experimentation, independent verification, and circuit prototyping  
- `RoadSelect/Quantum/`: Official implementation directly coupled with the main road selection workflow  

If organizing code appendices for a paper, it is recommended to describe them as "experimental version" and "engineering version" respectively.  

---  

## 8. Typical Input/Output File Instructions  

### 8.1 Input Files  
- `stroke{i}.txt`: Text file of road relationships after stroke aggregation  
- `nodedata{i}.txt`: Node attribute text file  
- `road_stroke.shp`: Stroke layer  
- `road_sample.shp`: Reference road network for comparison  
- `*.gdb`: ArcGIS File Geodatabase  

### 8.2 Intermediate Files  
- `Matrix/start.h5`: Initial state matrix  
- `Matrix/U_*.h5`: U operator matrix  
- `Matrix/Result/result_alpha_*.h5`: Evolution result blocks  

### 8.3 Final Results  
- `PageRankWeightResult.json`  
- `QPageRankWeightResult.json`  
- `GNNResult.json`  
- `selectresult.json`  
- `cover.json`  
- `connectivity.json`  
- `pearson_density.json`  

---  

## 9. How to Run Individual Modules  
If you only want to run a specific step, you can directly call the corresponding `run_*` function.  

### Example: Run Classic PageRank  
```python
from RoadSelect.Classic.PageRankWeightEngine import run_ClassicEngine
run_ClassicEngine(index)
```  

### Example: Run Quantum PageRank Main Workflow  
```python
from RoadSelect.Quantum.CreatStartEngine import run_CreatStartEngine
from RoadSelect.Quantum.CreatUEngine import run_CreatUEngine
from RoadSelect.Quantum.ApplyUEngine import run_ApplyUEngine
from RoadSelect.Quantum.ResultProcess import run_ResultProcess

run_CreatStartEngine(index)
run_CreatUEngine(index)
run_ApplyUEngine(index)
run_ResultProcess(index)
```  

### Example: Run GNN  
```python
from RoadSelect.GNN.GNN import runGNN
runGNN(index)
```  

### Example: Export Selected Road Results  
```python
from RoadSelect.Compare.SelectRate import run_SelectRate
run_SelectRate(index)
```  

---  

## 10. Notes  
- The `venv/` directory contains virtual environment files and is not recommended as the basis for running the project.  
- `Engine/QPageRankEngine.py` is more suitable as a "process control script" rather than the only entry point.  
- Some path notations depend on the current working directory; always run the code from the project root directory.  
- Part of the scripts target Windows/ArcGIS environments; ArcPy-related steps need to be replaced before migrating to Linux.  
