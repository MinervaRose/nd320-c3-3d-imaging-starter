#!/bin/bash

# Register the DICOM routing script with Orthanc
curl -X POST http://localhost:8042/tools/execute-script --data-binary @route_dicoms.lua -v

# Start the listener to receive routed DICOMs
sudo storescp 106 -v -aet HIPPOAI -od /home/workspace/src/routed --sort-on-study-uid st
