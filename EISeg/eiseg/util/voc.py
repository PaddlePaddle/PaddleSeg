import xml.etree.ElementTree as ET


class VocAnnotations:
    def __init__(self,
                 filename,
                 witdh,
                 height,
                 depth=3,
                 foldername="VOC2007",
                 sourcename="Unknown"):
        self.root = ET.Element("annotation")
        self.foleder = ET.SubElement(self.root, "folder")
        self.filename = ET.SubElement(self.root, "filename")
        self.source = ET.SubElement(self.root, "source")
        self.size = ET.SubElement(self.root, "size")
        self.width = ET.SubElement(self.size, "width")
        self.height = ET.SubElement(self.size, "height")
        self.depth = ET.SubElement(self.size, "depth")

        self.foleder.text = foldername
        self.filename.text = filename
        self.source.text = sourcename
        self.width.text = str(witdh)
        self.height.text = str(height)
        self.depth.text = str(depth)

    def savefile(self, filename):
        tree = ET.ElementTree(self.root)
        tree.write(filename, xml_declaration=False, encoding='utf-8')

    def add_object(self,
                   label_name,
                   xmin,
                   ymin,
                   xmax,
                   ymax,
                   tpose=0,
                   ttruncated=0,
                   tdifficult=0):
        object = ET.SubElement(self.root, "object")
        namen = ET.SubElement(object, "name")
        namen.text = label_name
        pose = ET.SubElement(object, "pose")
        pose.text = str(tpose)
        truncated = ET.SubElement(object, "truncated")
        truncated.text = str(ttruncated)
        difficult = ET.SubElement(object, "difficult")
        difficult.text = str(tdifficult)
        bndbox = ET.SubElement(object, "bndbox")
        xminn = ET.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = ET.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = ET.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = ET.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)
