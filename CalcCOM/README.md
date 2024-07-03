# Documentation- CalcCOM 

## Table of Contents 
| Description | Script |Functions |
| ------------- | ------------- | ------------- |
| Calculate **COM** | calc_centermass| |
| Overlay **Skeleton** on Video | displayskeleton| |
| Calc **segments with Deleva** | segdim_deleva| [segmentdim](#function-segmentdim) |

### End Table of Contents  <br/>

## Script: segdim_deleva
### Function segmentdim
[Link to Code: segdim_deleva](https://github.com/USCBiomechanicsLab/labcodes/blob/master/CalcCOM/segdim_deleva.py)

### **Keywords:**
Deleva, Segments, Mass, Length, Percent, Weight 


### **Syntax:**
```
from CalcCOM.segdim_deleva import segmentdim

segments = segmentdim(sex)                            
```
### Dependencies 
* **pd:** pandas  
* **np** numpy

### **Description:**<br/>
Create data frame containing segment center of mass length and percent weight
        based on de Leva 1996. 
[Link to Paper](https://www.sciencedirect.com/science/article/pii/0021929095001786)

de Leva. Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. 1996  
Based on digitized points:  
    segment:      origin                  -   other
    head:         vertex                  -   cervicale (C7)
    trunk:        cervical (C7)           -   mid-hip
    upperarm:     shoulder joint center   -   elbow joint center
    forearm:      elbow joint center      -   stylion (distal point radial styloid)
    hand:         sylion (see forearm)    -   3rd dactylion (tip of 3rd digit)
    thigh:        hip joint center        -   knee joint center 
    shank:        knee joint center       -   lateral malleolus
    foot:         heel                    -   acropodion (tip of longest toe - 1st or 2nd) 
   
 
### **Arguments:**

#### *Inputs*

   * **sex:** STR segmental parameters for sex ('f' or 'm')
   
#### *Outputs*

   * **segments:** DATAFRAME contains origin and other location for segment definition,
        proximal and distal joints (primarily for digitized point),
        as well as segmental center of mass position (cmpos),
        percent weight (massper), and sagittal radii of gyration (r_gyr).
   
### **Examples:**
Helpful examples

[Back to Table of Contents](#table-of-contents) 
