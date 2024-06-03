def get_center_of_bbox(bbox):
    """
    Get the center of the bounding box
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """
    Get the width of the bounding box
    """
    return bbox[2] - bbox[0]