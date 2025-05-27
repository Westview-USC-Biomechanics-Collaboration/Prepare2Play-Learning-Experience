class FileNameHelper:
    def __init__(self):
        pass

    def COMFileName(self, original_name: str) -> str:
        """
        Generate a COM file name based on the original file name.
        :param original_name: The original file name.
        :return: The new file name for the COM data.
        """
        return f"{original_name}_COM.mp4"