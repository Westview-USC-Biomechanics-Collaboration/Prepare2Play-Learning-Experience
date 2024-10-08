Imports DataServerLib

' This application creates the DaqControl using WithEvents to establish the
' event mechanism for updates on data acquisition progress.

Public Class Sample3
    ' Declare a member variable for the Daq control interface using WithEvents
    Private WithEvents daq As DaqControl
    Private prevScan As Integer

    ' The events are called in the DataServer's daq thread, so the local client
    ' must delegate to the local thread. Define the delegates accordingly.
    Delegate Sub AddDataDelegate()
    Delegate Sub CompleteDataDelegate()
    Private addDataDel As AddDataDelegate
    Private completeDataDel As CompleteDataDelegate

    Private Sub Form1_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load
        ' Initialize the button state
        StartBtn.Enabled = True
        StartBtn.Visible = True
        StopBtn.Enabled = False
        StopBtn.Visible = False

        ' declare a dataserver object, get a DaqControl object from server factory
        Dim server As New DataServer

        ' create object, assigning a configuration file, board type, and Instacal board number
        ' TODO:
        ' set path to XML configuration file
        ' set board type and MCC Instacal board number
        daq = server.CreateDaqObject("..\..\..\..\..\XML\config.xml", BoardType.brdKistler5691A1, 0)

        ' set the internal buffer to update every 50 ms
        Dim adv As IDaqAdvanced = daq
        adv.SetMCCParameters(1000, 50)
        adv.SetLocalParameters(64)

        ' Create delegates for event handling
        addDataDel = New AddDataDelegate(AddressOf AddData)
        completeDataDel = New CompleteDataDelegate(AddressOf CompleteData)

        ' enable measure
        daq.MeasureOn()

        ' read offsets
        daq.ReadOffsets(RangeType.rngBIP10VOLTS)
    End Sub

    Private Sub Form1_FormClosing(ByVal sender As System.Object, ByVal e As System.Windows.Forms.FormClosingEventArgs) Handles MyBase.FormClosing
        ' set to reset
        daq.MeasureOff()
    End Sub

    Private Sub Button1_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles StartBtn.Click
        prevScan = 0

        ' Start acquisition
        ' 1000Hz for 0 seconds (data acquisition occurs continuous mode)
        daq.Start(1000, 0, TriggerType.trigImmediate, RangeType.rngBIP10VOLTS)

        ' enable disable the start button, enable the stop button
        StartBtn.Enabled = False
        StartBtn.Visible = False
        StopBtn.Enabled = True
        StopBtn.Visible = True
    End Sub

    Private Sub Button2_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles StopBtn.Click
        daq.Stop()
    End Sub

    Public Sub AddData()
        Dim currScan As Integer = daq.LastAvailableScan

        If currScan <= prevScan Then
            Return
        End If

        ' get the device collection
        Dim dc As DeviceCollection = daq.GetDeviceCollection()

        ' get the plate from the device collection
        ' TODO: match the device name to the configuration file
        Dim plate As IPlate = dc.get_DeviceByName("LabAmp-5536327")

        ' get the shear and vertical force data
        ' TODO: arrTime takes same sample rate as specified above
        Dim Timearr As Single() = plate.arrTime(1000, prevScan, currScan)
        Dim Fxarr As Single() = plate.arrFx(prevScan, currScan)
        Dim Fyarr As Single() = plate.arrFy(prevScan, currScan)
        Dim Fzarr As Single() = plate.arrFz(prevScan, currScan)

        'For i = LBound(Fzarr) To UBound(Fzarr)
        '    Fzarr(i)
        'Next i

        ' for this example we update text boxes with most recent sample
        Time.Text = Timearr(UBound(Timearr))
        Fx.Text = Fxarr(UBound(Fxarr))
        Fy.Text = Fyarr(UBound(Fyarr))
        Fz.Text = Fzarr(UBound(Fzarr))

        prevScan = currScan
    End Sub 'AddData

    Public Sub CompleteData()
        ' perform one final add
        AddData()
        StartBtn.Enabled = True
        StartBtn.Visible = True
        StopBtn.Enabled = False
        StopBtn.Visible = False
    End Sub

    Private Sub On_Update(ByVal last_scan As UInteger) Handles daq.NewDataAvailable
        ' The event is called from the daq thread. Must use Invoke to run in the 
        ' local thread that owns the controls
        Invoke(addDataDel)
    End Sub

    Private Sub On_Complete() Handles daq.DaqComplete
        ' The event is called from the daq thread. Must use Invoke to run in the 
        ' local thread that owns the controls
        Invoke(completeDataDel)
    End Sub

End Class
