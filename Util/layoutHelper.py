def layoutHelper(num:int, orientation:str) ->int:
    """
    Args: num means the location out of 12
    Output: the pixel location of the center
    """
    width = 1320
    height =  1080

    if orientation=="vertical":
        return int(height * num / 12)
    elif orientation=="horizontal":
        return int(width * num / 12)
    else:
        return None