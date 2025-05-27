class COM_helper:
    def __init__(self, path:str=None):
        self.file_path = path
        self.columns = ['landmark_0_x', 'landmark_0_y', 'landmark_0_visibility', 'landmark_1_x', 'landmark_1_y', 'landmark_1_visibility', 'landmark_2_x', 'landmark_2_y', 'landmark_2_visibility', 'landmark_3_x', 'landmark_3_y', 'landmark_3_visibility', 'landmark_4_x', 'landmark_4_y', 'landmark_4_visibility', 'landmark_5_x', 'landmark_5_y', 'landmark_5_visibility', 'landmark_6_x', 'landmark_6_y', 'landmark_6_visibility', 'landmark_7_x', 'landmark_7_y', 'landmark_7_visibility', 'landmark_8_x', 'landmark_8_y', 'landmark_8_visibility', 'landmark_9_x', 'landmark_9_y', 'landmark_9_visibility', 'landmark_10_x', 'landmark_10_y', 'landmark_10_visibility', 'landmark_11_x', 'landmark_11_y', 'landmark_11_visibility', 'landmark_12_x', 'landmark_12_y', 'landmark_12_visibility', 'landmark_13_x', 'landmark_13_y', 'landmark_13_visibility', 'landmark_14_x', 'landmark_14_y', 'landmark_14_visibility', 'landmark_15_x', 'landmark_15_y', 'landmark_15_visibility', 'landmark_16_x', 'landmark_16_y', 'landmark_16_visibility', 'landmark_17_x', 'landmark_17_y', 'landmark_17_visibility', 'landmark_18_x', 'landmark_18_y', 'landmark_18_visibility', 'landmark_19_x', 'landmark_19_y', 'landmark_19_visibility', 'landmark_20_x', 'landmark_20_y', 'landmark_20_visibility', 'landmark_21_x', 'landmark_21_y', 'landmark_21_visibility', 'landmark_22_x', 'landmark_22_y', 'landmark_22_visibility', 'landmark_23_x', 'landmark_23_y', 'landmark_23_visibility', 'landmark_24_x', 'landmark_24_y', 'landmark_24_visibility', 'landmark_25_x', 'landmark_25_y', 'landmark_25_visibility', 'landmark_26_x', 'landmark_26_y', 'landmark_26_visibility', 'landmark_27_x', 'landmark_27_y', 'landmark_27_visibility', 'landmark_28_x', 'landmark_28_y', 'landmark_28_visibility', 'landmark_29_x', 'landmark_29_y', 'landmark_29_visibility', 'landmark_30_x', 'landmark_30_y', 'landmark_30_visibility', 'landmark_31_x', 'landmark_31_y', 'landmark_31_visibility', 'landmark_32_x', 'landmark_32_y', 'landmark_32_visibility', 'frame_index', 'COM_x', 'COM_y\n']
  

    def read_line(self, file_path, index):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    return self.__parse_line(line.split(","))
        raise IndexError("Line index out of range")

    
    def __parse_line(self, line):
        index = 0
        parsed_line = []
        while index < len(line)-3:
            item = line[index]
            parsed_line.append({"x":line[index],"y":line[index+1],"visibility":line[index+2]})
            index += 3
        
        parsed_line.append({"name":"COM","x":line[-2],"y":line[-1]})
        return parsed_line
    
    def drawFigure(self, row:int):
        linedata = self.read_line(self.file_path, row)

if __name__ == "__main__":
    com = COM_helper()
    line = com.read_line('coord.txt', 1)
    print(line)
