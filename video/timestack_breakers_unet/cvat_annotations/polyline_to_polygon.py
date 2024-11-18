#!/usr/bin/env python

import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
from xml.dom import minidom
import sys
from dunex.video.timestack_breakers_unet.cvat_annotations.cvat_xml import parse_polylines_by_image

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def polyline_to_polygon(points: List[Tuple[float, float]], width: float = 1.0) -> List[Tuple[float, float]]:
    """Convert a polyline to a polygon by giving it width."""
    if len(points) < 2:
        return []
    
    # Convert points to numpy array for easier calculation
    points = np.array(points)
    
    # Calculate vectors between consecutive points
    vectors = points[1:] - points[:-1]
    
    # Calculate normalized perpendicular vectors
    perp_vectors = np.zeros_like(vectors)
    perp_vectors[:, 0] = -vectors[:, 1]
    perp_vectors[:, 1] = vectors[:, 0]
    perp_vectors = np.array([normalize_vector(v) for v in perp_vectors])
    
    # Calculate the offset for each segment
    half_width = width / 2
    offsets = perp_vectors * half_width
    
    # Create the polygon points
    upper_points = []
    lower_points = []
    
    # Handle first point
    upper_points.append(tuple(points[0] + offsets[0]))
    lower_points.append(tuple(points[0] - offsets[0]))
    
    # Handle middle points
    for i in range(1, len(points) - 1):
        # Average the offset vectors for smooth transitions
        avg_offset = normalize_vector(offsets[i-1] + offsets[i]) * half_width
        upper_points.append(tuple(points[i] + avg_offset))
        lower_points.append(tuple(points[i] - avg_offset))
    
    # Handle last point
    upper_points.append(tuple(points[-1] + offsets[-1]))
    lower_points.append(tuple(points[-1] - offsets[-1]))
    
    # Combine points to form polygon (go up one side and down the other)
    polygon_points = upper_points + lower_points[::-1]
    
    return polygon_points

def create_xml_output(image_polygons: Dict[str, List[List[Tuple[float, float]]]]) -> str:
    """Create XML output with polygon annotations."""
    root = ET.Element('annotations')
    
    for image_name, data in image_polygons.items():
        image_elem = ET.SubElement(root, 'image')
        image_elem.set('name', image_name)
        image_elem.set('width', str(data['width']))
        image_elem.set('height', str(data['height']))
        
        for polygon_points in data['polygons']:
            polygon = ET.SubElement(image_elem, 'polygon')
            polygon.set('label', 'breaker')
            polygon.set('source', 'auto')
            polygon.set('occluded', '0')
            
            # Convert points to string format
            points_str = ';'.join(f'{x:.2f},{y:.2f}' for x, y in polygon_points)
            polygon.set('points', points_str)
            
            polygon.set('z_order', '0')
    
    # Pretty print the XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')
    return xml_str

def main(xml_content: str, line_width: float = 1.0) -> str:
    """Main function to process XML and create polygon annotations."""
    # Parse original polylines
    image_data = parse_polylines_by_image(xml_content)
    
    # Process each image
    result_data = {}
    for image_name, data in image_data.items():
        # Convert each polyline to a polygon
        polygons = []
        for polyline in data['polylines']:
            polygon = polyline_to_polygon(polyline, width=line_width)
            if polygon:
                polygons.append(polygon)
        
        # Store results
        result_data[image_name] = {
            'polygons': polygons,
            'width': data['width'],
            'height': data['height']
        }
        
        print(f"Processed {image_name}:")
        print(f"  Original polylines: {len(data['polylines'])}")
        print(f"  Generated polygons: {len(polygons)}")
    
    # Generate XML output
    return create_xml_output(result_data)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python polyline_to_python.py input_xml output_xml')
        exit(1)
    
    _, input_xml_file, output_xml_file = sys.argv
    
    # Read input XML
    with open(input_xml_file, 'r') as f:
        xml_content = f.read()

    # Process and generate new XML with thin polygons
    output_xml = main(xml_content, line_width=1.0)

    # Save output XML
    with open(output_xml_file, 'w') as f:
        f.write(output_xml)

    print(f"\nCreated {output_xml_file}")