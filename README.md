# Run ssd mobilenet original model "90" class with tensorflow lite



you need opencv and tensorflow lite libs to build this project
the tensorflow could be a subdirectory like they suggest in there website or just point to the includes

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

To run 
./main [path/to/model.tflite] [path/to/classes.txt] [input/video/or/camera] [path/to/output/video] 
