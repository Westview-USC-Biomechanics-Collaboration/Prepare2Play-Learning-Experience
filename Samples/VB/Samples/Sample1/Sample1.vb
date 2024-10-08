Imports DataServerLib

Public Class Sample1
    ' Declare a member variable for the Daq control interface
    Private daq As DaqControl

    Private Sub Form1_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load

        ' declare a dataserver object, get a DaqControl object from server factory
        Dim server As New DataServer

        ' create object, assigning a configuration file, board type, and Instacal board number
        ' TODO:
        ' set path to XML configuration file
        ' set board type and MCC Instacal board number
        daq = server.CreateDaqObject("..\..\..\..\..\XML\config.xml", BoardType.brdKistler5691A1, 0)

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
        ' Start acquisition
        ' 1000Hz for 5 seconds
        daq.Start(1000, 5000, TriggerType.trigImmediate, RangeType.rngBIP10VOLTS)

        ' Wait until done
        Do Until Not daq.Running()
        Loop
    End Sub
End Class
