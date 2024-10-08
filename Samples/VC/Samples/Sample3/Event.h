#include "stdafx.h"
#include <atlcom.h>	 // include for IDispEventImpl

// Define interface class for handling events
// User derive from this and implement handlers
class IEventAction
{
public:
	virtual void Initialize(DataServerLib::IDaqControlPtr DaqControlClass) = 0;
	virtual void ProcessData(int start_scan, int end_scan) = 0;
};


// Define event function description info structures
static _ATL_FUNC_INFO DaqCompleteInfo = {
         CC_STDCALL,	// Calling convention.
         VT_I4,			// Return type.
         0,				// Number of arguments.
         {}				// Argument types.
      };

static _ATL_FUNC_INFO NewDataAvailableInfo = {
         CC_STDCALL,	// Calling convention.
         VT_I4,			// Return type.
         1,				// Number of arguments.
         {VT_I4}		// Argument types.
      };

// Event handler class
template <class IEventActionImplType>
class CEvent : 
	public IDispEventImpl<0, CEvent<IEventActionImplType>, &__uuidof(DataServerLib::_IStatusEvents), &__uuidof(DataServerLib::__DataServerLib), 1, 0>
{
public:
	CEvent (DataServerLib::IDaqControlPtr DaqControlClass) : prevscan(0) {
		Advise (DaqControlClass);
		TheAction.Initialize(DaqControlClass);
	}
	
	~CEvent () {
		if( TheDaqControlClass != NULL ) {
			DispEventUnadvise (TheDaqControlClass);
		}
	}

	HRESULT Advise (DataServerLib::IDaqControlPtr DaqControlClass)
	{
		HRESULT hr = E_FAIL; 
		TheDaqControlClass = DaqControlClass;

		if( m_dwEventCookie == 0xFEFEFEFE ) { // don't advise if it's already been done
			hr = DispEventAdvise (TheDaqControlClass); 
		} 

		return hr;
	} 

	BEGIN_SINK_MAP(CEvent)
		SINK_ENTRY_INFO(0, __uuidof(DataServerLib::_IStatusEvents), 1, &CEvent::DaqComplete, &DaqCompleteInfo)
		SINK_ENTRY_INFO(0, __uuidof(DataServerLib::_IStatusEvents), 2, &CEvent::NewDataAvailable, &NewDataAvailableInfo)
	END_SINK_MAP()
	
	HRESULT __stdcall DaqComplete(void) {
		if( TheDaqControlClass != NULL ) {
			// check current amount available
			ULONG thisscan (TheDaqControlClass->LastAvailableScan); 

			if (thisscan>prevscan) {
				// call to process the data
				TheAction.ProcessData (prevscan, thisscan);
				prevscan = thisscan;
//				printf("complete! processed %d...\n", thisscan);
			}		
		}	
		return S_OK;
	}
	HRESULT __stdcall NewDataAvailable(ULONG last_scan) {
		if( TheDaqControlClass != NULL ) {
			
			// check current amount available
			ULONG thisscan (TheDaqControlClass->LastAvailableScan); 

			if (thisscan>prevscan) {
				// call to process the data
				TheAction.ProcessData (prevscan, thisscan);
				prevscan = thisscan;
//				printf("processed %d...\n", thisscan);
			}
		}
		return S_OK;
	}

protected:
	ULONG prevscan;
	DataServerLib::IDaqControlPtr	TheDaqControlClass;
	IEventActionImplType			TheAction;
};