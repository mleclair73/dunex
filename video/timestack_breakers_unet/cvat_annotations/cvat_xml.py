#!/usr/bin/env python

import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
from xml.dom import minidom
import sys

def parse_polylines_by_image(xml_content: str) -> Dict[str, Dict]:
    """Parse polyline points from XML content, grouped by image name."""
    root = ET.fromstring(xml_content)
    image_polylines = {}
    
    for image in root.findall('.//image'):
        image_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))
        polylines = []
        
        for polyline in image.findall('.//polyline'):
            points_str = polyline.get('points')
            points = [tuple(map(float, point.split(','))) 
                     for point in points_str.split(';')]
            polylines.append(points)
        
        if polylines:
            image_polylines[image_name] = {
                'polylines': polylines,
                'width': width,
                'height': height
            }
    
    return image_polylines


def parse_polygons_from_xml(xml_file: str) -> Dict[str, Dict]:
    """Parse polygon points from XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    image_polygons = {}
    
    for image in root.findall('.//image'):
        image_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))
        
        polygons = []
        for polygon in image.findall('.//polygon'):
            points_str = polygon.get('points')
            points = [tuple(map(float, point.split(','))) 
                     for point in points_str.split(';')]
            polygons.append(points)
            
        if polygons:
            image_polygons[image_name] = {
                'polygons': polygons,
                'width': width,
                'height': height
            }
    
    return image_polygons