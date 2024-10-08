# Prerequisite
# python -m pip install pywin32
# python -m pip install pywin32-ctypes

# When DataServer is installed it creates a registry entry for the COM dll identifying the CLSID
# The CLSID then references the dll location. We can load using either the Class name or the CLSID
# ex:  win32com.client.Dispatch("Dataserver.Dataserver") 
# or   win32com.client.Dispatch("{CCCA2D2D-0DAF-4C91-9F54-5044CF1ED72E}")

# Do this once to load typdef constants from COM object into the python cache
# import win32com.client
# xl = win32com.client.gencache.EnsureDispatch('DataServer.DataServer')
# C = win32com.client.constants

import win32com.client
from win32com.client import gencache

gencache.EnsureModule('{0F22982F-F8B2-4605-8F6D-2922770B9087}', 0, 1, 3)

# create the server object
server = win32com.client.Dispatch("Dataserver.Dataserver")

# Load the configuration file containing the devices
board_type = win32com.client.constants.brdKistlerEthernet
daq = server.CreateDaqObject("Config.xml", 3, 0)

# get the collection of devices
collection = daq.GetDeviceCollection()

# Dump information about the plate(s)
for plate in collection:
    # Get the IDeviceMetadata interface and list the details for the device
    metadata = win32com.client.CastTo(plate, 'IDeviceMetadata')
    print("Device Information:")
    print("Name: ", metadata.Name)
    print("Serial Number: ", metadata.SerialNumber)
    print("Manufacturer: ", metadata.Manufacturer)
    print("Model: ", metadata.Model)
    print("Start Channel: ", metadata.StartChannel)
    print("Nbr. of Channels: ", metadata.NumberOfChannels)
    print("Width (mm): ", metadata.Width_mm)
    print("Length (mm): ", metadata.Length_mm)
    print("Offset X (mm): ", metadata.dX_mm)
    print("Offset Y (mm): ", metadata.dY_mm)
    print("Angle (deg.): ", metadata.Alpha_degrees)
    print("Netowork Addr.: ", metadata.NetworkAddress)
    print("Data Server Version: ", metadata.DataServerVersion)
    if metadata.MoCapDataExists:
        print("MoCap dx (mm): ", metadata.Location_dx_mm)
        print("MoCap dy (mm): ", metadata.Location_dy_mm)
        print("MoCap dz (mm): ", metadata.Location_dz_mm)
        print("MoCap rx (deg.): ", metadata.Orientation_rx_deg)
        print("MoCap ry (deg.): ", metadata.Orientation_ry_deg)
        print("MoCap rz (deg.): ", metadata.Orientation_rz_deg)
    print("") # blank line <CR>

# Set amplifer to measure on
print("Measure On")
daq.MeasureOn()

# define some parameters
rate_per_channel = 1000
samples_per_channel = 10000
trigger = win32com.client.constants.trigImmediate
daq_range =  win32com.client.constants.rngBIP10VOLTS

# start daq
daq.Start(rate_per_channel, samples_per_channel, trigger, daq_range)

thisScan = -1
prevScan = 0

# Here, would could get a specific device by name 
# plate = collection.get_DeviceByName("LabAmp-5536327")
# But for now... we will dump info from all configured plates

while daq.Running:
    thisScan = daq.LastAvailableScan
    if thisScan > 0 and thisScan != prevScan:
        print("Processing from %d to %d..." % (prevScan, thisScan))
        for scan in range(prevScan, thisScan):
            print("[%5d]" % (scan), end=" ")        
            for plate in collection:
                print("Fx=%f Fy=%f Fz=%f" % (plate.Fx(scan), plate.Fx(scan), plate.Fz(scan)), end=" ")
            print("") # end line
        prevScan = thisScan
else:
    thisScan = daq.LastAvailableScan
    print("Finalize processing from %d to %d..." % (prevScan, thisScan))
    for scan in range(prevScan, thisScan+1):
        print("[%5d]  Fx=%f Fy=%f Fz=%f" % (scan, plate.Fx(scan), plate.Fx(scan), plate.Fz(scan)))
    
# Stop (should be stopped already)
daq.Stop()  

# Set amplifer to measure off
print("Measure Off")
daq.MeasureOff()






