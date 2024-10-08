// Sample1.cpp : A simple C++ example using DataServer.dll COM object.
//
// Instantiates DataServer, loads a configuration file, and acquires data.

#include "stdafx.h"
#include <objbase.h>

// import data server 
// TODO: change path to DataServer.dll
#import "C:\Projects\Biomechanics\DataServer\DataServer\DataServer\Release\DataServer.dll"

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
						
			// prepare data acqusition, and wait for software trigger event
			// Start (long rate_per_channel, long samples_per_channel, enum trigger_option, enum daq_range)
			daq->Start (1000, 5000, DataServerLib::trigImmediate, DataServerLib::rngBIP10VOLTS); 

			// loop while running
			while( daq->Running ) {
				// yield to other threads, let them do some work
				Sleep (50);
					
				// check current amount available
				ULONG thisscan (daq->LastAvailableScan); 

				if( thisscan>prevscan ) {
					// call to process the data
					// ProcessData (prevscan, thisscan);
					prevscan = thisscan;
					wprintf (L"processed %d...\n", thisscan);
				}
			}
					
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
}

