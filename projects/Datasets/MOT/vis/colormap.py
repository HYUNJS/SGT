from detectron2.utils.colormap import _COLORS


def id2color(id, rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = id % len(_COLORS)
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret

