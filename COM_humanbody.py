from CalcCOM.segdim_deleva import *
from CalcCOM.displayskeleton import addskeleton
from CalcCOM.calc_centermass import *
dataframe_male = segmentdim('m')
print(dataframe_male)
# main(sex= 'm',segments= dataframe_male)
#
addskeleton("data/derenBasketballTest1.mp4",file_vid_n='skeletonvideo.mp4',segments=dataframe_male)
