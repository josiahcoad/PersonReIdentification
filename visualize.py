# ------------ CODE IN THIS FILE FOR CSCE 625 ---------------
# pylint: disable=C0111
from os import listdir
from os.path import join

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import data
from args import args

DIRECTORY = "./predictions"


def print_grid(filepaths, avg_precisions, prediction_details, epoch, mAP):
    # files: shape = (10,10), each row is a query picture path  followed by the top 9 prediciton paths
    font_large = ImageFont.truetype("Anonymous_Pro.ttf", 160)
    font_medium = ImageFont.truetype("Anonymous_Pro.ttf", 80)
    font_small = ImageFont.truetype("Anonymous_Pro.ttf", 40)
    new_img = Image.new('RGB', (3000, 3100))
    for i in range(min(9, len(filepaths))):  # iter through rows
        for j in range(min(10, len(filepaths[i]))):  # iter through cols
            img = Image.open(filepaths[i][j])
            img = img.resize((140, 280), Image.ANTIALIAS)
            draw = ImageDraw.Draw(img)
            if j == 0:
                # if j == 0, we are looking at the query photo
                # draw the average precision on the picture
                ap = avg_precisions[i]
                draw.text((0, 0), str(
                    round(ap, 3)), (255, 255, 255), font=font_small)
                # outline the query picture in a scale from red for poor average precision to blue for excelent
                img = ImageOps.expand(img, border=6,
                                          fill=(int((1 - ap) * 255), 0, int(ap * 255)))
            else:
                # else, we are looking at one of the gallery photos...
                _, score, ismatch = prediction_details[i][j-1]
                # draw the "prediction score" on the picture
                draw.text((0, 0), str(round(1+score, 3)),
                          (255, 255, 255), font=font_small)
                # outline the picture in red if the ground truth is that the
                # picture is not the same identity as the query and blue if it is the same id
                img = ImageOps.expand(
                    img, border=6, fill='blue' if ismatch else 'red')
            new_img.paste(img, ((j)*300 + 50, (i+1)*300 + 50))
    draw = ImageDraw.Draw(new_img)
    # draw the Epoch and mAP to top of the new_img, add a label for the query column and draw a line
    draw.text((new_img.size[0] / 2 - 600, 80),
              f"Epoch: {epoch}, mAP: {round(float(mAP), 3)}", (255, 255, 255), font=font_large)
    draw.text((40, 260), "Query", font=font_medium, fill=(255, 0, 0, 255))
    draw.line((280, 0, 280, new_img.size[1]), fill=128, width=10)
    new_img.save(f"{DIRECTORY}/epoch{epoch}.png")


def get_epoch(filename):
    return int(filename.split("epoch")[1].split(".")[0])


def make_gif():
    images = []
    for filepath in sorted(filter(lambda x: x.endswith(".png"), listdir(DIRECTORY)), key=get_epoch):
        images.append(imageio.imread(join(DIRECTORY, filepath)))
    imageio.mimsave(f'{DIRECTORY}/training.gif', images, 'GIF', duration=.2)


def parse_pred(pred):
    idx, id_, score, ismatch = pred.split(',')
    return int(idx), int(id_), float(score), bool(int(ismatch))


def parse_line(line):
    query, test = line.split(':')
    query_idx, query_id, avg_precision = query.split(',')
    test_predictions = test.split('|')
    predictions = [parse_pred(pred) for pred in test_predictions]
    return int(query_idx), int(query_id), float(avg_precision), predictions


def build_image(filepath):
    with open(filepath) as file_:
        lines = file_.readlines()

    loader = data.Data(args)
    filepaths = []
    prediction_details = []
    avg_precisions = []
    query_predictions, mAP = np.array(lines)[:-1], lines[-1]
    for query in query_predictions:
        query_idx, _, avg_precision, predictions = parse_line(query)
        query_path = loader.queryset.imgs[query_idx]
        prediction_details.append([tuple(p[1:]) for p in predictions])
        files = [query_path] + [loader.testset.imgs[p[0]] for p in predictions]
        filepaths.append(files)
        avg_precisions.append(avg_precision)

    print_grid(filepaths, avg_precisions,
               prediction_details, get_epoch(filepath), mAP)


def main():
    for filename in listdir(DIRECTORY):
        if filename.endswith(".txt"):
            build_image(join(DIRECTORY, filename))
    make_gif()


if __name__ == "__main__":
    main()
