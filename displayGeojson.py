import re

bbox = "LngLat(-2.4303948053598106, 53.207615976982964),LngLat(-2.4283085791206815, 53.20636604677071)"
bboxLatLon = re.findall("\(.*?\)", bbox)
bboxList = []
for latLon in bboxLatLon:
    bboxList.append(
        latLon.replace("(", "").replace(")", "").replace(" ", "").split(",")
    )

print(bboxList)
