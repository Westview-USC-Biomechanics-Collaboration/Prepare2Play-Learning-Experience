using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

// set 'using' reference for DataServer library
// or prefix all calls with DataServerLib. 
// ex: DataServerLib.IDataServer = new DataServerLib.DataServer()
using DataServerLib;

// Add a reference to DataServer.dll COM object library.
// 1. right click on the project node in the solution explorer and select "Add Reference..."
// 2. select "COM" on the far left as the type of componenent you want to reference
// 3. select "Data Server 1.2 Type Library" componenet then click "Add"
// sa: http://msdn.microsoft.com/en-us/library/wkze6zky(v=vs.80).aspx

namespace Sample1
{
    class Program
    {
        // function to process data
        static void ProcessData(IPlate plate, int start_scan, int end_scan)
        {
            Console.WriteLine("processing from " + start_scan + " to " + end_scan + "...");

            // get the data from the plate interface
            System.Array fx = plate.arrFx(start_scan, end_scan);
            System.Array fy = plate.arrFy(start_scan, end_scan);
            System.Array fz = plate.arrFz(start_scan, end_scan);

            for (int i = fx.GetLowerBound(0); i <= fx.GetUpperBound(0); i++) {
                Console.WriteLine("[" + (start_scan + i) + "] " + fx.GetValue(i) + " " + fy.GetValue(i) + " " + fz.GetValue(i));
            }
            // or, use a ForEach
        }

        static void Main(string[] args)
        {
		    // local variable to track completion.
		    int prevscan = 0;
            
		    try
		    {
                // Create the server object
                IDataServer server = new DataServer();

			    // Get DAQ object, set Configuration file, board type, and Instacal Board Number
			    // TODO: 
			    // - change path to config.xml if needed. (parameter 1)
			    // - set board type (parameter 2)
			    // - set board number from InstaCal (parameter 3)
			    IDaqControl daq;
                daq = server.CreateDaqObject("../../../../../XML/config.xml", BoardType.brdKistler5695A1, 0);

			    // set amplifier to on 
			    daq.MeasureOn ();

			    // read offets if enabled
			    daq.ReadOffsets (RangeType.rngBIP10VOLTS);

			    IDeviceCollection dc = daq.GetDeviceCollection();

			    // Look up device by name (from the XML file)
			    // The generic device interface provides Time, and Voltage Data.
			    // the device name is the user defined name in XML file. Ex: <Name>Plate 1</Name>
			    // TODO: Set device name to match a plate in the XML config file.
			    IDevice device = dc.get_DeviceByName ("Plate 1");

    			// Get interface to plate
			    // The plate serves Fx, Fy, Fz, Mx, My, Mz, Ax, Ay, Time, and Voltage Data.
			    // Or, use IKistlerPlate to get raw force Fx12, Fx34, Fy14, Fy23, Fz1, Fz2, Fz3, Fz4 Data.
			    IPlate plate = (IPlate)device;
						
			    // prepare data acqusition, and wait for software trigger event
			    // Start (long rate_per_channel, long samples_per_channel, enum trigger_option, enum daq_range)
                daq.Start(100, 500, TriggerType.trigImmediate, RangeType.rngBIP10VOLTS); 

			    // loop while running
			    while( daq.Running ) {
				    // yield to other threads, let them do some work
				    System.Threading.Thread.Sleep (50);
					
				    // check current amount available
				    int thisscan = daq.LastAvailableScan; 

				    if( thisscan>prevscan ) {
                        // call to process the data
                        ProcessData(plate, prevscan, thisscan);

                        // set previous scan index
                        prevscan = thisscan;
				    }
			    }

                // process any remaining data
                ProcessData(plate, prevscan, daq.LastAvailableScan);

			    // call stop (although, it should already be stopped)
			    daq.Stop ();
					
			    // place plate into reset
			    daq.MeasureOff ();
		    }
		    catch( Exception ex ) 
		    { 
			    Console.WriteLine ("Unexpected COM exception: " + ex.Message);
		    }
        }
    }
}
