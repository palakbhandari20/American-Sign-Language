import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 28
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue

        cv2.putText(frame, 'Press "Q" to start collecting!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            print("Exiting data collection early.")
            break

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
