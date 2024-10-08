// Sample2.cpp : Defines the entry point for the console application.
//
//
// Extends Sample1.  Retrieves Data after acquisition.

#include "stdafx.h"
#include <objbase.h>
#include <atlsafe.h> // include for CComSafeArray

// import data server 
// TODO: change path to DataServer.dll
#import "C:\Projects\Biomechanics\DataServer\DataServer\DataServer\Release\DataServer.dll"

// function to process data
void ProcessData(DataServerLib::IPlatePtr plate, int start_scan, int end_scan)
{
	CComSafeArray<float> fx;
	fx.Attach (plate->arrFx (start_scan, end_scan));

	CComSafeArray<float> fy;
	fy.Attach (plate->arrFy (start_scan, end_scan));

	CComSafeArray<float> fz;
	fz.Attach (plate->arrFz (start_scan, end_scan));

	for (int i=fx.GetLowerBound(); i<=fx.GetUpperBound(); i++) {
		printf("[%d] %f, %f, %f\n", start_scan+i, fx[i], fy[i], fz[i]);
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	CoInitialize (NULL);
	
	// Build Scope for COM CoInitialize/CoUnitialize
	{ 
		// local variable to track completion.
		ULONG prevscan (0);
	
		// Create the server object
		DataServerLib::IDataServerPtr server;
		server.CreateInstance( L"DataServer.DataServer" );
		if( server==NULL )
			return -1;
		
		try
		{
			// Get DAQ object, set Configuration file, board type, and Instacal Board Number
			// TODO: 
			// - change path to config.xml if needed. (parameter 1)
			// - set board type (parameter 2)
			// - set board number from InstaCal (parameter 3)
			DataServerLib::IDaqControlPtr daq;
			daq = server->CreateDaqObject (L"../../../XML/config.xml", DataServerLib::brdKistler5691A1, 0);

			// set amplifier to on 
			daq->MeasureOn ();

			// read offets if enabled
			daq->ReadOffsets (DataServerLib::rngBIP10VOLTS);

			// Get the collection of devices
			DataServerLib::IDeviceCollectionPtr dc = daq->GetDeviceCollection();
			if( dc==NULL )
				return -1;

			// Look up device by name (from the XML file)
			// The generic device interface provides Time, and Voltage Data.
			// the device name is the user defined name in XML file. Ex: <Name>Plate 1</Name>
			// TODO: Set device name to match a plate in the XML config file.
			DataServerLib::IDevicePtr device = dc->get_DeviceByName (L"Plate 1");
			if( device==NULL )
				return -1;

			// Get interface to plate
			// The plate pointer servers Fx, Fy, Fz, Mx, My, Mz, Ax, Ay, Time, and Voltage Data.
			// Or, use IKistlerPlatePtr to get raw force Fx12, Fx34, Fy14, Fy23, Fz1, Fz2, Fz3, Fz4 Data.
			DataServerLib::IPlatePtr plate = device;
			if( plate==NULL )
				return -1;
						
			// prepare data acqusition, and wait for software trigger event
			// Start (long rate_per_channel, long samples_per_channel, enum trigger_option, enum daq_range)
			daq->Start (100, 500, DataServerLib::trigImmediate, DataServerLib::rngBIP10VOLTS); 

			// loop while running
			while( daq->Running ) {
				// yield to other threads, let them do some work
				Sleep (50);
					
				// check current amount available
				ULONG thisscan (daq->LastAvailableScan); 

				if( thisscan>prevscan ) {
					// call to process the data
					ProcessData (plate, prevscan, thisscan);
					prevscan = thisscan;
				}
			}

			// process any remaining data
			ProcessData (plate, prevscan, daq->LastAvailableScan);
								
			// call stop (although, it should already be stopped)
			daq->Stop ();
					
			// place plate into reset
			daq->MeasureOff ();
		}
		catch( _com_error e ) 
		{ 
			wprintf ((LPCTSTR)e.Description());
		}
	}

	CoUninitialize();
	return 0;
}

