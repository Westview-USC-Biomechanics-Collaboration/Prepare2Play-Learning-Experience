// Sample3.cpp : Defines the entry point for the console application.
//
// Extends Sample2.  Uses events to generate updates.

#include "stdafx.h"
#include <objbase.h>
#include <atlsafe.h> // include for CComSafeArray
#include <conio.h>	 // include for _kbhit()

// import data server 
// TODO: change path to DataServer.dll
#import "C:\Projects\Biomechanics\DataServer\DataServer\DataServer\Debug\DataServer.dll"

#include "event.h"	 // include for DataServer ConnectionPoint events

// Define a class to handle actions to process data.
// Must implement Initialize(...) and ProcessData(...) methods.
class MyHandler : public IEventAction
{
public:
	virtual void Initialize(DataServerLib::IDaqControlPtr DaqControlClass)
	{
		// Get the device collection
		DataServerLib::IDeviceCollectionPtr dc = DaqControlClass->GetDeviceCollection();

		// Look up plate device by name (from the XML file)
		// TODO: Set device name to match a plate in the XML config file.
		ThePlate = dc->get_DeviceByName (L"Plate 1");
	} 

	virtual void ProcessData(int start_scan, int end_scan)
	{
		// check if ThePlate is valid
		if( ThePlate == NULL ) 
			return;

		CComSafeArray<float> fx;
		fx.Attach (ThePlate->arrFx (start_scan, end_scan));

		CComSafeArray<float> fy;
		fy.Attach (ThePlate->arrFy (start_scan, end_scan));

		CComSafeArray<float> fz;
		fz.Attach (ThePlate->arrFz (start_scan, end_scan));

		for (int i=fx.GetLowerBound(); i<=fx.GetUpperBound(); i++) {
			printf("[%d] %f, %f, %f\n", start_scan+i, fx[i], fy[i], fz[i]);
		}
	}

protected:
	// This example is only interested in a single plate
	// Establish an interface to the object in the constructor, then use it until done.
	DataServerLib::IPlatePtr ThePlate;
};


int _tmain(int argc, _TCHAR* argv[])
{
	CoInitialize (NULL);
	
	// Build Scope for COM CoInitialize/CoUnitialize
	// This forces all auto-pointer objects to destroy prior to calling
	// CoUnitialize() - or else exceptions would be thrown.
	{ 
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

			// Setup a few advance DAQ parameters to enhance real-time 
			// ie: SetMCCParameters (buffer_length_ms, update_rate_ms);
			// for our example the MCC buffer is 2s, and updates every 50ms (approximately)
			DataServerLib::IDaqAdvancedPtr adv (daq);
			adv->SetMCCParameters (2000, 10);

			// set the size of the internal processing buffer (converting from MCC to voltages) 
			// smaller number will provide faster Real time updates
			// SetLocalParameters (long process_buffer_sz);
			// default is 1000
			adv->SetLocalParameters (64);

			CEvent<MyHandler> event_handler (daq);

			// set amplifier to on
			daq->MeasureOn ();

			// read offets if enabled
			daq->ReadOffsets (DataServerLib::rngBIP10VOLTS);
						
			// prepare data acqusition, and wait for software trigger event
			// Start (long rate_per_channel, long samples_per_channel, enum trigger_option, enum daq_range)
			// Note:
			// samples_per_channel = 0 -> will acquire data continuously until Stop() called.
			// trigger_option = trigSoftware -> setup and wait for trigger() function call.
			daq->Start (100, 0, DataServerLib::trigSoftware, DataServerLib::rngBIP10VOLTS); 

			// wait for key-press as trigger
			printf("press any key to start\n");
			while( !_kbhit() ) {}
			_getch();
			
			// GO!
			// FYI - could also use DataServerLib::trigImmediate above and it would have started
			daq->trigger();

			// loop while running
			printf("press any key to stop\n");

			while( daq->Running && !_kbhit() ) {
				// yield to other threads, let them do some work
				Sleep (10);
			}
			
			// Stop the acquisition process
			daq->Stop ();

			if( _kbhit () ) // clear the key-press
				_getch();
					
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

