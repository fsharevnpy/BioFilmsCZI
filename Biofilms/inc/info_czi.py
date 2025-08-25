from aicspylibczi import CziFile

def parse_czi_metadata_from_file(filepath):
    czi = CziFile(filepath)
    root = czi.meta  # Already an ElementTree.Element in your version

    def find_text(elem, path):
        tag = elem.find(path)
        return tag.text.strip() if tag is not None and tag.text else None

    result = {}

    # Image info
    image = root.find("Metadata/Information/Image")
    result['SizeX'] = find_text(image, "SizeX")
    result['SizeY'] = find_text(image, "SizeY")
    result['SizeZ'] = find_text(image, "SizeZ")
    result['SizeC'] = find_text(image, "SizeC")
    result['PixelType'] = find_text(image, "PixelType")

    # Pixel size
    result['PixelSize'] = {}
    for dist in root.findall("Metadata/Scaling/Items/Distance"):
        axis = dist.attrib.get("Id")
        value = find_text(dist, "Value")
        if axis and value:
            result['PixelSize'][axis] = float(value)

    # Channels
    result['Channels'] = []
    for ch in root.findall("Metadata/Information/Image/Dimensions/Channels/Channel"):
        result['Channels'].append({
            "Name": ch.attrib.get("Name"),
            "Fluor": find_text(ch, "Fluor"),
            "Excitation": find_text(ch, "ExcitationWavelength"),
            "Emission": find_text(ch, "EmissionWavelength"),
            "DetectionRange": find_text(ch.find("DetectionWavelength"), "Ranges") if ch.find("DetectionWavelength") is not None else None
        })

    # Objective info
    obj = root.find("Metadata/Information/Instrument/Objectives/Objective")
    result['Objective'] = {
        "Model": find_text(obj.find("Manufacturer"), "Model"),
        "NA": find_text(obj, "LensNA"),
        "Magnification": find_text(obj, "NominalMagnification"),
        "Immersion": find_text(obj, "Immersion")
    } if obj is not None else {}

    # Microscope system
    scope = root.find("Metadata/Information/Instrument/Microscopes/Microscope")
    result['Microscope'] = find_text(scope, "System") if scope is not None else None

    return result