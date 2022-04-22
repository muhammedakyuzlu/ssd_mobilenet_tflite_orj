# Run ssd mobilenet original model "90" classes with tensorflow lite



you need opencv and tensorflow lite libs to build this project
the tensorflow lite could be a subdirectory like they suggest in there website or
just point to the includes directory and libtensorflowlite.so (build with bazel) 

The tree structure of directory:
```md
├── CMakeLists.txt
├── include
│   ├── flatbuffers
│   └── tensorflow
|       |──core
|       |    └──utils
|       └──lite
├── lib
│   └── libtensorflowlite.so
└── src
│    └── *.h *.cpp
└── build
│    └── 
└── models
     └── ssd_mobilenet_90.tflite
     └── label_coco_90.txt

```
change the path in CMakeLists.txt

INCLUDE_DIRECTORIES to point to the include dir and 

set_property to point to lib/libtensorflowlite.so


create a build directory and run 

cmake .. && make

### To run ```python
./main [path/to/model.tflite] [path/to/classes.txt] [input/video/or/camera] [path/to/output/video] 
```
