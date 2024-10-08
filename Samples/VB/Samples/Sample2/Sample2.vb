Imports DataServerLib

' This application establishes a timer to poll the progress of data acquisition.

Public Class Sample2

    ' Declare a member variable for the Daq control interface
    Private daq As DaqControl
    Private prevScan As Integer

    Private Sub Form1_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load
        Dim UpdateInterval_ms As Integer = 50

        Timer1.Enabled = False
        Timer1.Interval = UpdateInterval_ms

        ' declare a dataserver object, get a DaqControl object from server factory
        Dim server As New DataServer

        ' create object, assigning a configuration file, board type, and Instacal board number
        ' TODO:
        ' set path to XML configuration file
        ' set board type and MCC Instacal board number
        daq = server.CreateDaqObject("..\..\..\..\..\XML\config.xml", BoardType.brdKistler5691A1, 0)

        ' set the internal buffer to update every 50 ms
        Dim adv As IDaqAdvanced = daq
        adv.SetMCCParameters(1000, UpdateInterval_ms)
        adv.SetLocalParameters(64)

        ' enable measure
        daq.MeasureOn()

        ' read offsets
        daq.ReadOffsets(RangeType.rngBIP10VOLTS)
    End Sub

    Private Sub Form1_FormClosing(ByVal sender As System.Object, ByVal e As System.Windows.Forms.FormClosingEventArgs) Handles MyBase.FormClosing
        ' set to reset
        daq.MeasureOff()
    End Sub

    Private Sub Button1_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button1.Click
        prevScan = 0

        ' Start acquisition
        ' 1000Hz for 5 seconds
        daq.Start(1000, 5000, TriggerType.trigImmediate, RangeType.rngBIP10VOLTS)

        ' enable the time, disable the start button
        Timer1.Enabled = True
        Button1.Enabled = False
        Button1.Visible = False
    End Sub

    Public Sub AddData()
        Dim currScan As Integer = daq.LastAvailableScan
        Dim size As Integer = currScan - prevScan

        If size = 0 Then
            Return
        End If

        ' get the device collection
        Dim dc As DeviceCollection = daq.GetDeviceCollection()

        ' get the plate from the device collection
        ' TODO: match the device name to the configuration file
        Dim plate As IPlate = dc.get_DeviceByName("Plate 1")

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

    ' Timer runs in GUI thread
    Private Sub Timer1_Tick(ByVal sender As Object, ByVal e As System.EventArgs) Handles Timer1.Tick
        AddData()

        If (Not daq.Running) Then
            Button1.Enabled = True
            Button1.Visible = True
            Timer1.Enabled = False
        End If
    End Sub

End Class