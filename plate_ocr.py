from easyocr import Reader

def read_plate(file_path):
    reader = Reader(['en'])

    result = reader.readtext(file_path)
    print(result)

    # plate_num = "".join([c if c != "|" else "1" for c in result[1][-2] if c.isalnum() or c == "|"])
    plate_num = "".join([c if c != "|" else "1" for c in result[0][-2] if c.isalnum() or c == "|"])

    return plate_num

if __name__ == "__main__":
    print(read_plate('./license_plate_image/my-plate.png'))