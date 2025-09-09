import cv2

class View:
    @staticmethod
    def showWorkout(img, workout, fps):
        img = cv2.flip(img, 1)

        # Counter box
        cv2.rectangle(img, (12, 6), (425, 100), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'Count: {int(workout.count)}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        # Percentage + progress bar
        color = (0, 255, 0) if (workout.per >= 95 or workout.per <= 5) else (0, 0, 255)
        cv2.putText(img, f'{int(workout.per)}%', (1200, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.rectangle(img, (1200, 100), (1275, 650), (0, 0, 255), 3)
        cv2.rectangle(img, (1200, int(workout.bar)), (1275, 650), (0, 0, 255), cv2.FILLED)

        # FPS
        cv2.putText(img, f'FPS: {int(fps)}', (20, 730),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

        return img

    @staticmethod
    def showInstruction(img):
        img = cv2.flip(img, 1)
        cv2.rectangle(img, (430, 740), (1335, 620), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, 'Take your position.', (440, 710),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        return img
