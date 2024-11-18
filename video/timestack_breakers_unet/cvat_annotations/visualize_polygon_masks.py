#!/usr/bin/env python
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import distinctipy
import sys
from cvat_annotations.cvat_xml import parse_polygons_from_xml

def create_boolean_mask(polygons: List[List[Tuple[float, float]]], 
                       width: int, 
                       height: int) -> np.ndarray:
    """Create a boolean mask where polygon areas are True and background is False."""
    # Create empty uint8 mask (cv2 compatible)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw each polygon
    for polygon in polygons:
        # Convert points to integer array
        points = np.array(polygon, dtype=np.int32)
        
        # Draw filled polygon with 1
        cv2.fillPoly(mask, [points], 255)
    
    return mask

def create_colored_mask(polygons: List[List[Tuple[float, float]]], 
                       width: int, 
                       height: int) -> np.ndarray:
    """Create a colored mask where each polygon has a distinct color."""
    # Create empty RGB mask
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get distinct colors for each polygon
    n_colors = len(polygons)
    colors = distinctipy.get_colors(n_colors)
    
    # Convert colors to BGR (OpenCV format) and scale to 0-255
    colors_bgr = [(int(b * 255), int(g * 255), int(r * 255)) 
                  for r, g, b in colors]
    
    # Draw each polygon with its distinct color
    for polygon, color in zip(polygons, colors_bgr):
        # Convert points to integer array
        points = np.array(polygon, dtype=np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(mask, [points], color)
    
    return mask

def main(xml_file: str, output_dir: str = 'visualization') -> None:
    """Main function to create colored visualizations of polygons."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse polygons from XML
    image_data = parse_polygons_from_xml(xml_file)
    
    print(f"Found {len(image_data)} images with polygons")
    
    # Process each image
    for image_name, data in image_data.items():
        # Create colored mask
        mask = create_colored_mask(
            data['polygons'], 
            data['width'], 
            data['height']
        )
        
        # Save visualization
        output_name = f"viz_{image_name}"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, mask)
        
        print(f"\nProcessed {image_name}:")
        print(f"  Number of polygons: {len(data['polygons'])}")
        print(f"  Output saved as: {output_name}")

# Generate visualizations
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python visualize_polygon_masks.py input_xml output_dir')
        exit(1)
    
    _, input_xml_file, output_dir = sys.argv
    

    main(input_xml_file, output_dir)