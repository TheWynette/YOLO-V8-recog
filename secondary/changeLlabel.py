import os
import glob
import xml.etree.ElementTree as ET


def convert_annotation(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find("bndbox")
        b = (
            int(xml_box.find("xmin").text),
            int(xml_box.find("ymin").text),
            int(xml_box.find("xmax").text),
            int(xml_box.find("ymax").text),
        )
        image_width = int(root.find("size").find("width").text)
        image_height = int(root.find("size").find("height").text)

        # YOLO format: <object-class> <x_center> <y_center> <width> <height>
        x_center = (b[0] + b[2]) / (2 * image_width)
        y_center = (b[1] + b[3]) / (2 * image_height)
        width = (b[2] - b[0]) / image_width
        height = (b[3] - b[1]) / image_height

        yolo_lines.append(f"{cls_id} {x_center} {y_center} {width} {height}")

    return yolo_lines


def batch_convert_xml_to_yolo(xml_folder, output_folder, classes):
    os.makedirs(output_folder, exist_ok=True)
    xml_files = glob.glob(os.path.join(xml_folder, "*.xml"))

    for xml_file in xml_files:
        yolo_lines = convert_annotation(xml_file, classes)
        output_file = os.path.join(
            output_folder, os.path.splitext(os.path.basename(xml_file))[0] + ".txt"
        )

        with open(output_file, "w") as f:
            for line in yolo_lines:
                f.write(line + "\n")


if __name__ == "__main__":
    xml_folder = (
        r"C:\Users\marsh\Desktop\recognition\yolo\labelled_xml"  # 输入你的XML文件夹路径
    )
    output_folder = r"C:\Users\marsh\Desktop\recognition\yolo\labelled_yolo"  # 输入你的输出文件夹路径
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # 输入你的类别列表

    batch_convert_xml_to_yolo(xml_folder, output_folder, classes)
