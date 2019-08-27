from PIL import ImageDraw, ImageFont


def draw_bb_on_img(faces, img):
    draw = ImageDraw.Draw(img)
    fs = max(20, round(img.size[0] * img.size[1] * 0.000005))
    for face in faces:
        draw.rectangle(((int(face.bb.left), int(face.bb.top)), (int(face.bb.right), int(face.bb.bottom))),
                       outline='green', width=5)
        draw.text(
            (int(face.bb.left), int(face.bb.bottom) + 20),
            "%s %.2f%%" % (face.top_prediction.name.upper(), face.top_prediction.confidence * 100),
            font=ImageFont.truetype('fonts/font.ttf', fs)
        )
