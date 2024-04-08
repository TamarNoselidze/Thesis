from google.cloud import vision

import os
import fnmatch


def detect_labels(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)   # by default will return 10 labels it detects
    # response = client.label_detection(image=image, max_results=5)   # to specify how many labels we want
    labels = response.label_annotations

    for label in labels:
        print(f"Label: {label.description}, Confidence: {label.score}")


def find_jpeg_files(directory_path):
    jpeg_files = []
    for root, dirnames, filenames in os.walk(directory_path):
        for filename in fnmatch.filter(filenames, '*.JPEG'):
            jpeg_files.append(os.path.join(root, filename))
    return jpeg_files

def find_perturbed_files(directory_path):
    png_files = []
    for root, dirnames, filenames in os.walk(directory_path):
        for filename in fnmatch.filter(filenames, '*.png'):
            png_files.append(os.path.join(root, filename))
    return png_files


if __name__ == '__main__':
    # image_path = './khinkali.jpg'

    directory_path = './TREMBA/dataset/Imagenet/Sample_10/'
    jpeg_files = find_jpeg_files(directory_path)
    output_path = './TREMBA/output/'
    perturbed_files = find_perturbed_files(output_path)


    # image_path = './film_bakur.jpeg'
    # detect_labels(image_path)


    for i in range(len(jpeg_files)-1):

        image_path = jpeg_files[i]

        perturbed_image_path = perturbed_files[i]

        print("labels for the original image:")
        detect_labels(image_path)
        print("\nlabels for the perturbed image:")
        detect_labels(perturbed_image_path)
        print("="*80)

