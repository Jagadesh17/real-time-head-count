import cv2
import numpy as np

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        self.tracked_students = {}
        self.max_disappeared = 10  # Tolerance for the number of frames a student can disappear
        self.disappeared = {}
        self.next_student_id = 0

    def __del__(self):
        self.video.release()

    def register(self, centroid):
        self.tracked_students[self.next_student_id] = centroid
        self.disappeared[self.next_student_id] = 0
        self.next_student_id += 1

    def deregister(self, student_id):
        del self.tracked_students[student_id]
        del self.disappeared[student_id]

    def update(self, faces):
        if len(faces) == 0:
            for student_id in list(self.disappeared.keys()):
                self.disappeared[student_id] += 1

                if self.disappeared[student_id] > self.max_disappeared:
                    self.deregister(student_id)
            return self.tracked_students

        input_centroids = np.zeros((len(faces), 2), dtype="int")

        for (i, (x, y, w, h)) in enumerate(faces):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.tracked_students) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            existing_ids = list(self.tracked_students.keys())
            students_centroids = np.array(list(self.tracked_students.values()))

            distances = np.linalg.norm(students_centroids - input_centroids[:, np.newaxis], axis=2)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if distances[row, col] < 50:
                    student_id = existing_ids[col]
                    self.tracked_students[student_id] = input_centroids[row]
                    self.disappeared[student_id] = 0

                    used_rows.add(row)
                    used_cols.add(col)

            unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distances.shape[1])).difference(used_cols)

            if distances.shape[0] >= distances.shape[1]:
                for row in unused_rows:
                    self.register(input_centroids[row])

            for col in unused_cols:
                student_id = existing_ids[col]
                self.disappeared[student_id] += 1

                if self.disappeared[student_id] > self.max_disappeared:
                    self.deregister(student_id)

        return self.tracked_students

    def get_frame(self):
        success, image = self.video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        self.update(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(image, f'Students: {len(self.tracked_students)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
